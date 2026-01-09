use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub name: String,
    pub emoji: String,
    pub distance: f32,
}

#[derive(Debug, Clone)]
pub struct PopularityStats {
    pub emoji: String,
    pub clicks: i64,
    pub shown: i64,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path.as_ref()).context("Failed to open database")?;
        Ok(Self { conn })
    }

    pub fn init_schema(&self, _embedding_dim: usize) -> Result<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emojis (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                sanitized TEXT NOT NULL,
                emoji TEXT NOT NULL
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emoji_embeddings (
                id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )",
            [],
        )?;
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS popularity (
                query TEXT NOT NULL,
                emoji TEXT NOT NULL,
                shown INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                PRIMARY KEY (query, emoji)
            )",
            [],
        )?;
        Ok(())
    }

    pub fn insert_emoji(
        &self,
        name: &str,
        sanitized: &str,
        emoji: &str,
        embedding: &[f32],
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO emojis (name, sanitized, emoji) VALUES (?1, ?2, ?3)",
            params![name, sanitized, emoji],
        )?;
        let rowid = self.conn.last_insert_rowid();
        let embedding_bytes = serialize_f32_vec(embedding);
        self.conn.execute(
            "INSERT INTO emoji_embeddings(id, embedding) VALUES (?1, ?2)",
            params![rowid, embedding_bytes],
        )?;
        Ok(rowid)
    }

    pub fn search_similar(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut stmt = self.conn.prepare("SELECT e.id, e.name, e.emoji, v.embedding FROM emojis e JOIN emoji_embeddings v ON e.id = v.id")?;
        let mut results: Vec<SearchResult> = Vec::new();
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let emoji: String = row.get(2)?;
            let embedding_bytes: Vec<u8> = row.get(3)?;
            Ok((id, name, emoji, embedding_bytes))
        })?;

        for row in rows {
            let (_, name, emoji, embedding_bytes) = row?;
            let embedding = deserialize_f32_vec(&embedding_bytes);
            let distance = 1.0 - cosine_similarity(query_embedding, &embedding);
            results.push(SearchResult {
                name,
                emoji,
                distance,
            });
        }

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
        Ok(results)
    }

    pub fn get_popularity(&self, query: &str) -> Result<HashMap<String, (i64, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT emoji, clicks, shown FROM popularity WHERE query = ?1")?;
        let results = stmt.query_map(params![query], |row| {
            Ok((
                row.get::<_, String>(0)?,
                (row.get::<_, i64>(1)?, row.get::<_, i64>(2)?),
            ))
        })?;
        let mut map = HashMap::new();
        for result in results {
            let (emoji, stats) = result?;
            map.insert(emoji, stats);
        }
        Ok(map)
    }

    pub fn update_popularity(&self, query: &str, emoji: &str, event: &str) -> Result<()> {
        let (shown_inc, clicks_inc) = match event {
            "shown" => (1, 0),
            "click" => (0, 1),
            _ => (0, 0),
        };
        self.conn.execute(
            "INSERT INTO popularity(query, emoji, shown, clicks) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(query, emoji) DO UPDATE SET shown = shown + ?3, clicks = clicks + ?4",
            params![query, emoji, shown_inc, clicks_inc],
        )?;
        Ok(())
    }

    pub fn get_stats(&self, limit: usize) -> Result<Vec<PopularityStats>> {
        let mut stmt = self.conn.prepare(
            "SELECT emoji, SUM(clicks) as total_clicks, SUM(shown) as total_shown FROM popularity
             GROUP BY emoji HAVING total_clicks != 0 ORDER BY total_clicks DESC LIMIT ?1",
        )?;
        let results = stmt.query_map(params![limit], |row| {
            Ok(PopularityStats {
                emoji: row.get(0)?,
                clicks: row.get(1)?,
                shown: row.get(2)?,
            })
        })?;
        results
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to collect stats")
    }

    pub fn get_top_queries(&self, emoji: &str, limit: usize) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT query, clicks FROM popularity WHERE emoji = ?1 AND clicks != 0 ORDER BY clicks DESC LIMIT ?2"
        )?;
        let results =
            stmt.query_map(params![emoji, limit], |row| Ok((row.get(0)?, row.get(1)?)))?;
        results
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to collect top queries")
    }

    pub fn begin_transaction(&mut self) -> Result<()> {
        self.conn.execute("BEGIN TRANSACTION", [])?;
        Ok(())
    }

    pub fn commit_transaction(&mut self) -> Result<()> {
        self.conn.execute("COMMIT", [])?;
        Ok(())
    }
}

fn serialize_f32_vec(vec: &[f32]) -> Vec<u8> {
    vec.iter().flat_map(|&f| f.to_le_bytes()).collect()
}

fn deserialize_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_f32_vec() {
        let vec = vec![1.0f32, 2.0f32, 3.0f32];
        let bytes = serialize_f32_vec(&vec);
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn test_deserialize_f32_vec() {
        let vec = vec![1.0f32, 2.0f32, 3.0f32];
        let bytes = serialize_f32_vec(&vec);
        let deserialized = deserialize_f32_vec(&bytes);
        assert_eq!(vec, deserialized);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&c, &d) - 0.0).abs() < 1e-6);
    }
}
