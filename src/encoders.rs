use anyhow::Result;
use ndarray::{Array1, Array2};
use std::path::Path;
use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

type OnnxModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub trait Encoder: Send + Sync {
    fn encode(&self, text: &str, is_query: bool) -> Result<Vec<f32>>;
    fn encode_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|text| self.encode(text, is_query))
            .collect()
    }
    fn embedding_dim(&self) -> usize;
    fn model_name(&self) -> &str;
}

fn mean_pooling(token_embeddings: &Array2<f32>, attention_mask: &[i64]) -> Array1<f32> {
    let (seq_len, dim) = (token_embeddings.shape()[0], token_embeddings.shape()[1]);
    let mut sum_embeddings = Array1::<f32>::zeros(dim);
    let mut sum_mask = 0.0f32;

    for i in 0..seq_len {
        let mask_val = attention_mask[i] as f32;
        if mask_val > 0.0 {
            for j in 0..dim {
                sum_embeddings[j] += token_embeddings[[i, j]] * mask_val;
            }
            sum_mask += mask_val;
        }
    }
    if sum_mask > 0.0 {
        sum_embeddings.mapv_inplace(|x| x / sum_mask);
    }
    sum_embeddings
}

fn normalize(embedding: Array1<f32>) -> Vec<f32> {
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        embedding.to_vec()
    }
}

pub struct MiniLMEncoder {
    tokenizer: Tokenizer,
    model: OnnxModel,
    embedding_dim: usize,
}

impl MiniLMEncoder {
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self {
            tokenizer,
            model,
            embedding_dim: 384,
        })
    }
}

impl Encoder for MiniLMEncoder {
    fn encode_batch(&self, texts: &[String], _is_query: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let (max_length, batch_size) = (128, texts.len());
        let mut all_input_ids = Vec::with_capacity(batch_size * max_length);
        let mut all_attention_masks = Vec::with_capacity(batch_size * max_length);

        for text in texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            let mut input_ids = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();
            if input_ids.len() < max_length {
                input_ids.resize(max_length, 0);
                attention_mask.resize(max_length, 0);
            } else {
                input_ids.truncate(max_length);
                attention_mask.truncate(max_length);
            }
            all_input_ids.extend(input_ids.iter().map(|&x| x as i64));
            all_attention_masks.extend(attention_mask.iter().map(|&x| x as i64));
        }

        let input_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((batch_size, max_length), all_input_ids.clone())?
                .into_dyn();
        let attention_mask_tensor = tract_ndarray::Array2::from_shape_vec(
            (batch_size, max_length),
            all_attention_masks.clone(),
        )?
        .into_dyn();
        let token_type_ids_tensor = tract_ndarray::Array2::from_shape_vec(
            (batch_size, max_length),
            vec![0i64; batch_size * max_length],
        )?
        .into_dyn();

        let outputs = self.model.run(tvec!(
            Tensor::from(input_ids_tensor).into(),
            Tensor::from(attention_mask_tensor).into(),
            Tensor::from(token_type_ids_tensor).into(),
        ))?;

        let output_tensor = outputs[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix3>()?;
        let (seq_len, hidden_dim) = (output_tensor.shape()[1], output_tensor.shape()[2]);

        let mut results = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let mut output_2d = Array2::<f32>::zeros((seq_len, hidden_dim));
            for i in 0..seq_len {
                for j in 0..hidden_dim {
                    output_2d[[i, j]] = output_tensor[[batch_idx, i, j]];
                }
            }
            let attention_mask =
                &all_attention_masks[batch_idx * max_length..(batch_idx + 1) * max_length];
            results.push(normalize(mean_pooling(&output_2d, attention_mask)));
        }
        Ok(results)
    }

    fn encode(&self, text: &str, _is_query: bool) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let max_length = 128;
        let mut input_ids = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();

        if input_ids.len() < max_length {
            input_ids.resize(max_length, 0);
            attention_mask.resize(max_length, 0);
        } else {
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
        }

        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
        let input_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), input_ids_i64.clone())?
                .into_dyn();
        let attention_mask_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), attention_mask_i64.clone())?
                .into_dyn();
        let token_type_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), vec![0i64; max_length])?
                .into_dyn();

        let outputs = self.model.run(tvec!(
            Tensor::from(input_ids_tensor).into(),
            Tensor::from(attention_mask_tensor).into(),
            Tensor::from(token_type_ids_tensor).into(),
        ))?;

        let output_tensor = outputs[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix3>()?;
        let (seq_len, hidden_dim) = (output_tensor.shape()[1], output_tensor.shape()[2]);
        let mut output_2d = Array2::<f32>::zeros((seq_len, hidden_dim));
        for i in 0..seq_len {
            for j in 0..hidden_dim {
                output_2d[[i, j]] = output_tensor[[0, i, j]];
            }
        }
        Ok(normalize(mean_pooling(&output_2d, &attention_mask_i64)))
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    fn model_name(&self) -> &str {
        "minilm"
    }
}

pub struct MultilingualMiniLMEncoder {
    tokenizer: Tokenizer,
    model: OnnxModel,
    embedding_dim: usize,
}

impl MultilingualMiniLMEncoder {
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self {
            tokenizer,
            model,
            embedding_dim: 384,
        })
    }
}

impl Encoder for MultilingualMiniLMEncoder {
    fn encode(&self, text: &str, _is_query: bool) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let max_length = 128;
        let mut input_ids = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();

        if input_ids.len() < max_length {
            input_ids.resize(max_length, 0);
            attention_mask.resize(max_length, 0);
        } else {
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
        }

        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
        let input_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), input_ids_i64.clone())?
                .into_dyn();
        let attention_mask_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), attention_mask_i64.clone())?
                .into_dyn();
        let token_type_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), vec![0i64; max_length])?
                .into_dyn();

        let outputs = self.model.run(tvec!(
            Tensor::from(input_ids_tensor).into(),
            Tensor::from(attention_mask_tensor).into(),
            Tensor::from(token_type_ids_tensor).into(),
        ))?;

        let output_tensor = outputs[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix3>()?;
        let (seq_len, hidden_dim) = (output_tensor.shape()[1], output_tensor.shape()[2]);
        let mut output_2d = Array2::<f32>::zeros((seq_len, hidden_dim));
        for i in 0..seq_len {
            for j in 0..hidden_dim {
                output_2d[[i, j]] = output_tensor[[0, i, j]];
            }
        }
        Ok(normalize(mean_pooling(&output_2d, &attention_mask_i64)))
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    fn model_name(&self) -> &str {
        "multilingual-minilm"
    }
}

pub struct E5Encoder {
    tokenizer: Tokenizer,
    model: OnnxModel,
    query_prefix: String,
    document_prefix: String,
    embedding_dim: usize,
}

impl E5Encoder {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
        query_prefix: String,
        document_prefix: String,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self {
            tokenizer,
            model,
            query_prefix,
            document_prefix,
            embedding_dim: 768,
        })
    }
}

impl Encoder for E5Encoder {
    fn encode(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        let prefix = if is_query {
            &self.query_prefix
        } else {
            &self.document_prefix
        };
        let prefixed_text = format!("{}{}", prefix, text);
        let encoding = self
            .tokenizer
            .encode(prefixed_text.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let max_length = 512;
        let mut input_ids = encoding.get_ids().to_vec();
        let mut attention_mask = encoding.get_attention_mask().to_vec();

        if input_ids.len() < max_length {
            input_ids.resize(max_length, 0);
            attention_mask.resize(max_length, 0);
        } else {
            input_ids.truncate(max_length);
            attention_mask.truncate(max_length);
        }

        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
        let input_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), input_ids_i64.clone())?
                .into_dyn();
        let attention_mask_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), attention_mask_i64.clone())?
                .into_dyn();
        let token_type_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, max_length), vec![0i64; max_length])?
                .into_dyn();

        let outputs = self.model.run(tvec!(
            Tensor::from(input_ids_tensor).into(),
            Tensor::from(attention_mask_tensor).into(),
            Tensor::from(token_type_ids_tensor).into(),
        ))?;

        let output_tensor = outputs[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<tract_ndarray::Ix3>()?;
        let (seq_len, hidden_dim) = (output_tensor.shape()[1], output_tensor.shape()[2]);
        let mut output_2d = Array2::<f32>::zeros((seq_len, hidden_dim));
        for i in 0..seq_len {
            for j in 0..hidden_dim {
                output_2d[[i, j]] = output_tensor[[0, i, j]];
            }
        }
        Ok(normalize(mean_pooling(&output_2d, &attention_mask_i64)))
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    fn model_name(&self) -> &str {
        "e5"
    }
}

pub fn create_encoder<P: AsRef<Path>>(
    encoder_type: &str,
    model_path: P,
    tokenizer_path: P,
) -> Result<Box<dyn Encoder>> {
    match encoder_type.to_lowercase().as_str() {
        "minilm" => Ok(Box::new(MiniLMEncoder::new(model_path, tokenizer_path)?)),
        "multilingual-minilm" => Ok(Box::new(MultilingualMiniLMEncoder::new(
            model_path,
            tokenizer_path,
        )?)),
        "e5" => Ok(Box::new(E5Encoder::new(
            model_path,
            tokenizer_path,
            "query: ".to_string(),
            "passage: ".to_string(),
        )?)),
        _ => Err(anyhow::anyhow!(
            "Unknown encoder type: {}. Supported types: minilm, multilingual-minilm, e5",
            encoder_type
        )),
    }
}
