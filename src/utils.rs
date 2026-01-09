use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use inquire::Select;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn download_file<P: AsRef<Path>>(url: &str, path: P, label: &str) -> Result<()> {
    println!("Downloading {}...", label);
    let response = reqwest::blocking::get(url).context("Download failed")?;
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut file = File::create(path.as_ref())?;
    let content = response.bytes()?;

    for (i, chunk) in content.chunks(8192).enumerate() {
        file.write_all(chunk)?;
        pb.set_position((i * 8192).min(total_size as usize) as u64);
    }

    pb.finish_with_message(format!(
        "Downloaded {} ({:.2} MB)",
        label,
        total_size as f64 / 1024.0 / 1024.0
    ));
    Ok(())
}

pub fn prompt_download(label: &str) -> Result<bool> {
    Ok(Select::new(
        &format!("{} not found. Download it?", label),
        vec!["yes", "no"],
    )
    .with_starting_cursor(0)
    .prompt()?
        == "yes")
}

pub fn sanitize(s: &str) -> String {
    s.to_lowercase()
        .replace('_', " ")
        .replace(':', "")
        .trim()
        .to_string()
}

pub fn distance_to_similarity(distance: f32) -> f32 {
    1.0 / (1.0 + distance)
}

pub fn calculate_hybrid_score(
    distance: f32,
    clicks: i64,
    shown: i64,
    rank: usize,
    alpha: f32,
) -> f32 {
    let cosine_sim = distance_to_similarity(distance);
    let ctr = (clicks + 1) as f32 / (shown + 2) as f32;
    let trust = ((shown - 5).max(0) as f32 / 5.0).min(1.0);
    let discount = 1.0 / (rank as f32 + 2.0).log2();
    let adjusted_ctr = ctr * (1.0 + discount);
    (1.0 - trust) * cosine_sim + trust * (alpha * cosine_sim + (1.0 - alpha) * adjusted_ctr)
}
