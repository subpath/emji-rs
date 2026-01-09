mod config;
mod database;
mod encoders;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use cli_clipboard::{ClipboardContext, ClipboardProvider};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use inquire::Select;
use std::collections::HashMap;
use std::fs;

use config::{Config, ModelSpec};
use database::{Database, SearchResult};
use encoders::create_encoder;
use utils::{calculate_hybrid_score, distance_to_similarity, prompt_download, sanitize};

#[derive(Parser)]
#[command(name = "emji")]
#[command(about = "Emoji semantic search CLI 🔍", long_about = None)]
struct Cli {
    #[arg(value_name = "QUERY")]
    query: Vec<String>,
    #[arg(short, long, default_value = "3")]
    n: usize,
    #[arg(long)]
    build_index: bool,
    #[arg(long)]
    cleanup: bool,
    #[arg(long)]
    show_stats: bool,
    #[arg(long)]
    list_models: bool,
    #[arg(long, value_name = "MODEL")]
    use_model: Option<String>,
    #[arg(long)]
    show_model: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = Config::config_path()?;

    if !config_path.exists() && !cli.list_models && cli.use_model.is_none() {
        prompt_model_selection()?;
    }

    if cli.list_models {
        return list_models();
    }
    if let Some(model_name) = cli.use_model {
        return switch_model(&model_name);
    }
    if cli.show_model {
        return show_current_model();
    }
    if cli.show_stats {
        return show_stats(cli.n);
    }
    if cli.cleanup {
        return cleanup();
    }
    if cli.build_index {
        ensure_dependencies()?;
        return build_index();
    }

    if cli.query.is_empty() {
        eprintln!("Error: No query provided\n");
        eprintln!("Usage: emji <QUERY>...");
        eprintln!("       emji --build-index");
        eprintln!("       emji --cleanup");
        eprintln!("       emji --show-stats");
        eprintln!("       emji --list-models");
        eprintln!("       emji --use-model <MODEL>");
        eprintln!("       emji --show-model");
        std::process::exit(1);
    }

    ensure_dependencies()?;
    query_emoji(&cli.query.join(" "), cli.n)?;
    Ok(())
}

fn ensure_dependencies() -> Result<()> {
    let config = Config::load(Config::config_path()?)?;
    let deps = vec![
        (Config::model_path()?, &config.model_url, "Model"),
        (
            Config::tokenizer_path()?,
            &config.tokenizer_url,
            "Tokenizer",
        ),
        (
            Config::default_emoji_path()?,
            &config.emoji_url,
            "Emoji data",
        ),
    ];

    for (path, url, label) in deps {
        if !path.exists() {
            if prompt_download(label)? {
                utils::download_file(url, &path, label)?;
            } else {
                anyhow::bail!("Cannot continue without {}. Exiting.", label);
            }
        }
    }
    Ok(())
}

fn build_index() -> Result<()> {
    let config = Config::load(Config::config_path()?)?;
    let db_path = Config::db_path()?;

    if db_path.exists() {
        let answer = Select::new(
            &format!("{} exists. Rebuild?", db_path.display()),
            vec!["yes", "no"],
        )
        .with_starting_cursor(0)
        .prompt()?;
        if answer == "yes" {
            fs::remove_file(&db_path)?;
        } else {
            return Ok(());
        }
    }

    let model_path = Config::model_path()?;
    let tokenizer_path = Config::tokenizer_path()?;
    let encoder = create_encoder(&config.encoder_type, &model_path, &tokenizer_path)?;
    println!(
        "Building embeddings using {} encoder...",
        encoder.model_name()
    );

    let emoji_file = Config::emoji_file()?;
    let emoji_json = fs::read_to_string(&emoji_file).context("Failed to read emoji file")?;
    let emojis: HashMap<String, String> =
        serde_json::from_str(&emoji_json).context("Failed to parse emoji JSON")?;

    let mut db = Database::new(&db_path)?;
    db.init_schema(encoder.embedding_dim())?;
    db.begin_transaction()?;

    let emoji_vec: Vec<(String, String)> = emojis.into_iter().collect();
    let pb = ProgressBar::new(emoji_vec.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    const BATCH_SIZE: usize = 32;
    println!("⚡ Generating embeddings in batches of {}...", BATCH_SIZE);

    let mut all_embeddings = Vec::with_capacity(emoji_vec.len());
    for chunk in emoji_vec.chunks(BATCH_SIZE) {
        let batch_texts: Vec<String> = chunk.iter().map(|(name, _)| sanitize(name)).collect();
        let batch_embeddings = encoder.encode_batch(&batch_texts, false)?;
        for (i, (name, symbol)) in chunk.iter().enumerate() {
            all_embeddings.push((
                name.clone(),
                batch_texts[i].clone(),
                symbol.clone(),
                batch_embeddings[i].clone(),
            ));
        }
        pb.inc(chunk.len() as u64);
    }

    pb.finish_and_clear();
    println!(
        "💾 Inserting {} embeddings into database...",
        all_embeddings.len()
    );

    let insert_pb = ProgressBar::new(all_embeddings.len() as u64);
    insert_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for (name, sanitized, symbol, embedding) in all_embeddings {
        db.insert_emoji(&name, &sanitized, &symbol, &embedding)?;
        insert_pb.inc(1);
    }

    db.commit_transaction()?;
    insert_pb.finish_with_message(format!("✅ Index built with {} entries", emoji_vec.len()));
    println!("Index saved to: {}", db_path.display());
    Ok(())
}

fn query_emoji(query: &str, n: usize) -> Result<()> {
    let config = Config::load(Config::config_path()?)?;
    let db_path = Config::db_path()?;

    if !db_path.exists() {
        let answer = Select::new("Emoji index not found. Build it now?", vec!["yes", "no"])
            .with_starting_cursor(0)
            .prompt()?;
        if answer == "yes" {
            build_index()?;
        } else {
            println!("Exiting. Index required for search.");
            return Ok(());
        }
    }

    let model_path = Config::model_path()?;
    let tokenizer_path = Config::tokenizer_path()?;
    let encoder = create_encoder(&config.encoder_type, &model_path, &tokenizer_path)?;

    let sanitized_query = sanitize(query);
    let query_embedding = encoder.encode(&sanitized_query, true)?;

    let db = Database::new(&db_path)?;
    let results = db.search_similar(&query_embedding, n * 2)?;

    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    let popularity = db.get_popularity(query)?;
    let mut scored_results: Vec<(SearchResult, f32, usize)> = results
        .iter()
        .enumerate()
        .map(|(rank, result)| {
            let (clicks, shown) = popularity.get(&result.emoji).copied().unwrap_or((0, 0));
            let score = calculate_hybrid_score(result.distance, clicks, shown, rank, config.alpha);
            (result.clone(), score, rank)
        })
        .collect();

    scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored_results.truncate(n);
    let final_results: Vec<SearchResult> = scored_results
        .into_iter()
        .map(|(result, _, _)| result)
        .collect();

    for result in &final_results {
        db.update_popularity(query, &result.emoji, "shown")?;
    }

    let choices: Vec<String> = final_results
        .iter()
        .map(|r| {
            format!(
                "{} {} {}",
                r.emoji,
                r.name,
                format!("{:.3}", distance_to_similarity(r.distance)).dimmed()
            )
        })
        .collect();

    let selected = Select::new("Select the correct emoji:", choices.clone())
        .with_starting_cursor(0)
        .prompt()?;
    let chosen_index = choices.iter().position(|c| c == &selected).unwrap();
    let chosen_emoji = &final_results[chosen_index].emoji;

    db.update_popularity(query, chosen_emoji, "click")?;

    let mut ctx = ClipboardContext::new()
        .map_err(|e| anyhow::anyhow!("Failed to access clipboard: {}", e))?;
    ctx.set_contents(chosen_emoji.to_string())
        .map_err(|e| anyhow::anyhow!("Failed to copy to clipboard: {}", e))?;
    println!("{} {} copied to clipboard", chosen_emoji, "✓".green());
    Ok(())
}

fn show_stats(n: usize) -> Result<()> {
    let db_path = Config::db_path()?;
    if !db_path.exists() {
        println!("No index found. Build it first.");
        return Ok(());
    }

    let db = Database::new(&db_path)?;
    let stats = db.get_stats(n)?;
    if stats.is_empty() {
        println!("No usage statistics available yet.");
        return Ok(());
    }

    println!("\n{}", "📊 Emoji Popularity Statistics".bold().cyan());
    println!("{}", "─".repeat(80));
    println!(
        "{:<8} {:<8} {:<8} {}",
        "Emoji".bold(),
        "Clicks".bold(),
        "Shown".bold(),
        "Top Queries".bold()
    );
    println!("{}", "─".repeat(80));

    for stat in stats {
        let top_queries = db.get_top_queries(&stat.emoji, 3)?;
        let queries_str: Vec<String> = top_queries
            .iter()
            .map(|(q, c)| format!("{} ({})", q, c))
            .collect();
        println!(
            "{:<8} {:<8} {:<8} {}",
            stat.emoji,
            stat.clicks,
            stat.shown,
            queries_str.join(", ")
        );
    }
    println!("{}", "─".repeat(80));
    Ok(())
}

fn prompt_model_selection() -> Result<()> {
    println!("\n{}", "🎉 Welcome to Emji!".bold().cyan());
    println!("{}", "─".repeat(80));
    println!("\nFirst, let's choose an embedding model for semantic emoji search.\n");
    println!(
        "💡 Recommendation: Start with {} for fast performance",
        "minilm".bold()
    );
    println!("   You can change models later using: emji --use-model <MODEL>\n");

    let registry = Config::get_model_registry();
    let mut model_list: Vec<(String, ModelSpec)> = registry.into_iter().collect();
    model_list.sort_by(|a, b| a.0.cmp(&b.0));

    let choices: Vec<String> = model_list
        .iter()
        .map(|(name, spec)| {
            if name == "minilm" {
                format!(
                    "{:<15} ~{}MB  - {} {}",
                    name,
                    spec.model_size_mb,
                    spec.description,
                    "(Recommended)".green()
                )
            } else {
                format!(
                    "{:<15} ~{}MB   - {}",
                    name, spec.model_size_mb, spec.description
                )
            }
        })
        .collect();

    let selection = Select::new("Select an embedding model:", choices)
        .with_starting_cursor(0)
        .prompt()?;
    let model_name = selection
        .split(|c: char| c.is_whitespace())
        .next()
        .unwrap_or("minilm");
    let spec = model_list
        .iter()
        .find(|(name, _)| name == model_name)
        .map(|(_, spec)| spec)
        .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

    let config = Config::from_model_spec(spec);
    config.save(&Config::config_path()?)?;

    println!("\n{} Model set to: {}", "✓".green(), model_name.bold());
    println!("   Encoder: {}", spec.encoder_type);
    println!("   Embedding dimension: {}", spec.embedding_dim);
    println!("\n💡 You can change models later using: emji --use-model <MODEL>\n");
    Ok(())
}

fn list_models() -> Result<()> {
    println!("\n{}", "📦 Available Embedding Models".bold().cyan());
    println!("{}", "─".repeat(80));
    println!("{:<15} {}", "Model".bold(), "Description".bold());
    println!("{}", "─".repeat(80));

    for (name, description) in Config::list_models() {
        println!("{:<15} {}", name, description);
    }

    println!("{}", "─".repeat(80));
    println!("\n💡 Usage: emji --use-model <MODEL>");
    println!("   Example: emji --use-model e5-base\n");
    Ok(())
}

fn switch_model(model_name: &str) -> Result<()> {
    let config_path = Config::config_path()?;
    let mut config = Config::load(&config_path)?;

    let registry = Config::get_model_registry();
    let spec = registry.get(model_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown model: {}. Use --list-models to see available models.",
            model_name
        )
    })?;

    config.switch_model(model_name)?;
    config.save(&config_path)?;

    println!("{} Switched to model: {}", "✓".green(), model_name.bold());
    println!("   Encoder: {}", spec.encoder_type);
    println!("   Embedding dimension: {}", spec.embedding_dim);
    println!("   Description: {}", spec.description);
    println!(
        "\n{} You need to rebuild the index for this model:",
        "⚠".yellow()
    );
    println!("   emji --build-index\n");
    Ok(())
}

fn show_current_model() -> Result<()> {
    let config = Config::load(Config::config_path()?)?;
    let registry = Config::get_model_registry();
    let spec = registry.get(&config.model_name);

    println!("\n{}", "🔧 Current Model Configuration".bold().cyan());
    println!("{}", "─".repeat(80));
    println!("Model name: {}", config.model_name.bold());
    println!("Encoder type: {}", config.encoder_type);

    if let Some(spec) = spec {
        println!("Embedding dimension: {}", spec.embedding_dim);
        println!("Description: {}", spec.description);
    }

    println!("Model URL: {}", config.model_url);
    println!("Tokenizer URL: {}", config.tokenizer_url);
    println!("{}", "─".repeat(80));

    let model_path = Config::model_path()?;
    let tokenizer_path = Config::tokenizer_path()?;
    let db_path = Config::db_path()?;

    println!("\n{}", "📁 File Status".bold());
    println!(
        "  Model: {}",
        if model_path.exists() {
            "✓ Downloaded".green()
        } else {
            "✗ Not downloaded".red()
        }
    );
    println!(
        "  Tokenizer: {}",
        if tokenizer_path.exists() {
            "✓ Downloaded".green()
        } else {
            "✗ Not downloaded".red()
        }
    );
    println!(
        "  Index: {}",
        if db_path.exists() {
            "✓ Built".green()
        } else {
            "✗ Not built".red()
        }
    );
    println!();
    Ok(())
}

fn cleanup() -> Result<()> {
    let home_dir = Config::home_dir()?;
    let answer = Select::new(
        &format!(
            "Delete all files in {}? This cannot be undone.",
            home_dir.display()
        ),
        vec!["yes", "no"],
    )
    .with_starting_cursor(1)
    .prompt()?;

    if answer == "yes" {
        fs::remove_dir_all(&home_dir)?;
        println!("🧹 Removed {}", home_dir.display());
    } else {
        println!("Cleanup aborted.");
    }
    Ok(())
}
