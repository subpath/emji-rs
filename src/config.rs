use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub encoder_type: String,
    pub model_url: String,
    pub tokenizer_url: String,
    pub embedding_dim: usize,
    pub model_size_mb: usize,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "ALPHA")]
    pub alpha: f32,

    #[serde(rename = "MODEL_NAME", default = "default_model_name")]
    pub model_name: String,

    #[serde(rename = "ENCODER_TYPE")]
    pub encoder_type: String,

    #[serde(rename = "MODEL_URL")]
    pub model_url: String,

    #[serde(rename = "TOKENIZER_URL")]
    pub tokenizer_url: String,

    #[serde(rename = "EMOJI_URL")]
    pub emoji_url: String,
}

fn default_model_name() -> String {
    "minilm".to_string()
}

impl Default for Config {
    fn default() -> Self {
        let default_model = Self::get_model_registry().get("minilm").unwrap().clone();
        Self::from_model_spec(&default_model)
    }
}

impl ModelSpec {
    pub fn new(
        name: &str,
        encoder_type: &str,
        model_url: &str,
        tokenizer_url: &str,
        embedding_dim: usize,
        model_size_mb: usize,
        description: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            encoder_type: encoder_type.to_string(),
            model_url: model_url.to_string(),
            tokenizer_url: tokenizer_url.to_string(),
            embedding_dim,
            model_size_mb,
            description: description.to_string(),
        }
    }
}

impl Config {
    pub fn get_model_registry() -> HashMap<String, ModelSpec> {
        let mut registry = HashMap::new();
        registry.insert("minilm".to_string(), ModelSpec::new(
            "minilm", "minilm",
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_qint8_arm64.onnx",
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
            384, 22, "Fast and lightweight - best for general use",
        ));
        registry.insert("minilm-l12".to_string(), ModelSpec::new(
            "minilm-l12", "minilm",
            "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/onnx/model.onnx",
            "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/tokenizer.json",
            384, 120, "Higher quality, slightly slower",
        ));
        registry.insert("multilingual".to_string(), ModelSpec::new(
            "multilingual", "multilingual-minilm",
            "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model.onnx",
            "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/tokenizer.json",
            384, 470, "Supports 50+ languages",
        ));
        registry
    }

    pub fn from_model_spec(spec: &ModelSpec) -> Self {
        Self {
            alpha: 0.2,
            model_name: spec.name.clone(),
            encoder_type: spec.encoder_type.clone(),
            model_url: spec.model_url.clone(),
            tokenizer_url: spec.tokenizer_url.clone(),
            emoji_url: "https://gist.githubusercontent.com/subpath/13bd5c15f76f451dfcb85421a53f0666/raw/1d362e4b4addfcd920b88f949090c6e82bf2c791/emojies_shortnames.json".to_string(),
        }
    }

    pub fn list_models() -> Vec<(String, String)> {
        let registry = Self::get_model_registry();
        let mut models: Vec<(String, String)> = registry
            .iter()
            .map(|(name, spec)| (name.clone(), spec.description.clone()))
            .collect();
        models.sort_by(|a, b| a.0.cmp(&b.0));
        models
    }

    pub fn switch_model(&mut self, model_name: &str) -> Result<()> {
        let registry = Self::get_model_registry();
        let spec = registry
            .get(model_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_name))?;
        self.model_name = spec.name.clone();
        self.encoder_type = spec.encoder_type.clone();
        self.model_url = spec.model_url.clone();
        self.tokenizer_url = spec.tokenizer_url.clone();
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            let config = Self::default();
            config.save(path)?;
            return Ok(config);
        }

        let contents = fs::read_to_string(path).context("Failed to read config file")?;
        let mut config: Config =
            serde_json::from_str(&contents).context("Failed to parse config file")?;

        if config.model_name.is_empty() {
            config.model_name = "minilm".to_string();
            config.save(path)?;
        }
        if config.encoder_type.is_empty() {
            config.encoder_type = "minilm".to_string();
            config.save(path)?;
        }
        Ok(config)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = serde_json::to_string_pretty(self).context("Failed to serialize config")?;
        fs::write(path, contents).context("Failed to write config file")?;
        Ok(())
    }

    pub fn home_dir() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
        let emji_home = home.join(".emji");
        if !emji_home.exists() {
            fs::create_dir_all(&emji_home).context("Failed to create .emji directory")?;
        }
        Ok(emji_home)
    }

    pub fn config_path() -> Result<PathBuf> {
        Ok(Self::home_dir()?.join(".config"))
    }

    pub fn model_path_for(model_name: &str) -> Result<PathBuf> {
        Ok(Self::home_dir()?.join(format!("model_{}.onnx", model_name)))
    }

    pub fn model_path() -> Result<PathBuf> {
        let config = Self::load(Self::config_path()?)?;
        Self::model_path_for(&config.model_name)
    }

    pub fn tokenizer_path_for(model_name: &str) -> Result<PathBuf> {
        Ok(Self::home_dir()?.join(format!("tokenizer_{}.json", model_name)))
    }

    pub fn tokenizer_path() -> Result<PathBuf> {
        let config = Self::load(Self::config_path()?)?;
        Self::tokenizer_path_for(&config.model_name)
    }

    pub fn db_path_for(model_name: &str) -> Result<PathBuf> {
        Ok(Self::home_dir()?.join(format!("emoji_index_{}.db", model_name)))
    }

    pub fn default_emoji_path() -> Result<PathBuf> {
        Ok(Self::home_dir()?.join("shortnames.json"))
    }

    pub fn override_emoji_path() -> Result<PathBuf> {
        Ok(Self::home_dir()?.join("shortnames_override.json"))
    }

    pub fn db_path() -> Result<PathBuf> {
        let config = Self::load(Self::config_path()?)?;
        Self::db_path_for(&config.model_name)
    }

    pub fn emoji_file() -> Result<PathBuf> {
        let override_path = Self::override_emoji_path()?;
        if override_path.exists() {
            Ok(override_path)
        } else {
            Self::default_emoji_path()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.alpha, 0.2);
        assert_eq!(config.encoder_type, "minilm");
    }

    #[test]
    fn test_save_and_load_config() -> Result<()> {
        let dir = tempdir()?;
        let config_path = dir.path().join("test_config.json");

        let config = Config::default();
        config.save(&config_path)?;

        let loaded = Config::load(&config_path)?;
        assert_eq!(config.alpha, loaded.alpha);
        assert_eq!(config.encoder_type, loaded.encoder_type);

        Ok(())
    }
}
