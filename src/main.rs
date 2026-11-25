use std::fs::File;

use anyhow::{Context, Result};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::data_type::ByteArray;
use parquet::file::statistics::Statistics;

/// Describe a parquet file and pretty-print sample rows.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Path to the parquet file.
    path: std::path::PathBuf,

    /// Number of sample rows to pretty-print.
    #[arg(short = 'n', long, default_value_t = 5)]
    rows: usize,

    /// Sample-row columns to print. Repeatable and comma-delimited; prints
    /// every column when absent.
    #[arg(short = 'c', long, value_delimiter = ',', alias = "column")]
    columns: Vec<String>,
}

fn run(cli: &Cli) -> Result<()> {
    let file =
        File::open(&cli.path).with_context(|| format!("failed to open {}", cli.path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).with_context(|| {
        format!(
            "failed to read parquet metadata from {}",
            cli.path.display()
        )
    })?;

    let metadata = builder.metadata();
    let schema_descr = metadata.file_metadata().schema_descr();
    let sample_rows = cli.rows.max(1);

    println!("file:       {}", cli.path.display());
    println!("rows:       {}", metadata.file_metadata().num_rows());
    println!("row groups: {}", metadata.num_row_groups());

    println!("schema ({} columns):", schema_descr.num_columns());
    let name_width = (0..schema_descr.num_columns())
        .map(|i| schema_descr.column(i).name().len())
        .max()
        .unwrap_or(0);
    let type_width = (0..schema_descr.num_columns())
        .map(|i| schema_descr.column(i).physical_type().to_string().len())
        .max()
        .unwrap_or(0);
    for i in 0..schema_descr.num_columns() {
        let col = schema_descr.column(i);
        let logical = match col.logical_type_ref() {
            Some(logical) => format!("{logical:?}"),
            None => "-".to_string(),
        };
        println!(
            "  {:<name_width$}  {:<type_width$}  {}",
            col.name(),
            col.physical_type(),
            logical
        );
    }

    for (i, group) in (0..metadata.num_row_groups())
        .map(|i| metadata.row_group(i))
        .enumerate()
    {
        let total_compressed = (0..group.num_columns())
            .map(|c| group.column(c).compressed_size())
            .sum::<i64>();
        let codecs = (0..group.num_columns())
            .map(|c| group.column(c).compression())
            .collect::<CompressionCounter>()
            .describe();
        println!(
            "  row group {i}: {} rows, {total_compressed} bytes compressed, {codecs}",
            group.num_rows(),
        );
    }

    // Per-column stats from the first row group to sanity-check a file.
    if metadata.num_row_groups() > 0 {
        let group = metadata.row_group(0);
        for i in 0..group.num_columns() {
            let Some(stats) = group.column(i).statistics() else {
                continue;
            };
            let desc = match stats {
                Statistics::Boolean(s) => format!(
                    "min={:?} max={:?} nulls={}",
                    s.min_opt(),
                    s.max_opt(),
                    nulls(s.null_count_opt())
                ),
                Statistics::Int32(s) => format!(
                    "min={:?} max={:?} nulls={}",
                    s.min_opt(),
                    s.max_opt(),
                    nulls(s.null_count_opt())
                ),
                Statistics::Int64(s) => format!(
                    "min={:?} max={:?} nulls={}",
                    s.min_opt(),
                    s.max_opt(),
                    nulls(s.null_count_opt())
                ),
                Statistics::Float(s) => format!(
                    "min={:?} max={:?} nulls={}",
                    s.min_opt(),
                    s.max_opt(),
                    nulls(s.null_count_opt())
                ),
                Statistics::Double(s) => format!(
                    "min={:?} max={:?} nulls={}",
                    s.min_opt(),
                    s.max_opt(),
                    nulls(s.null_count_opt())
                ),
                Statistics::ByteArray(s) => format!(
                    "min={} max={} nulls={}",
                    opt_bytes(s.min_opt().map(ByteArray::data)),
                    opt_bytes(s.max_opt().map(ByteArray::data)),
                    nulls(s.null_count_opt())
                ),
                Statistics::FixedLenByteArray(s) => format!(
                    "min={} max={} nulls={}",
                    opt_bytes(s.min_opt().map(|b| b.data())),
                    opt_bytes(s.max_opt().map(|b| b.data())),
                    nulls(s.null_count_opt())
                ),
                Statistics::Int96(s) => format!(
                    "min={:?} max={:?} nulls={}",
                    s.min_opt(),
                    s.max_opt(),
                    nulls(s.null_count_opt())
                ),
            };
            println!("  stats ({}): {desc}", schema_descr.column(i).name());
        }
    }

    let reader = builder
        .with_batch_size(sample_rows)
        .build()
        .context("failed to build record batch reader")?;

    let mut remaining = sample_rows;
    let mut sample = Vec::new();
    for batch in reader {
        let batch = batch.context("failed to read record batch")?;
        if batch.num_rows() == 0 {
            continue;
        }
        if batch.num_rows() <= remaining {
            remaining -= batch.num_rows();
            sample.push(batch);
        } else {
            sample.push(batch.slice(0, remaining));
            remaining = 0;
        }
        if remaining == 0 {
            break;
        }
    }

    let total = sample.iter().map(|b| b.num_rows()).sum::<usize>();
    println!("\n{sample_rows} sample rows (truncated):");
    if total == 0 {
        println!("(file has no rows)");
        return Ok(());
    }

    let options = FormatOptions::default();
    for (batch_no, batch) in sample.iter().enumerate() {
        let schema = batch.schema();
        let formatters = (0..batch.num_columns())
            .map(|c| {
                ArrayFormatter::try_new(batch.column(c), &options)
                    .with_context(|| format!("failed to format column {}", schema.field(c).name()))
            })
            .collect::<Result<Vec<_>>>()?;
        let wanted = if cli.columns.is_empty() {
            (0..batch.num_columns()).collect::<Vec<_>>()
        } else {
            cli.columns
                .iter()
                .map(|name| {
                    schema.index_of(name).with_context(|| {
                        format!(
                            "column {name:?} not found in {}; available columns: {}",
                            cli.path.display(),
                            schema
                                .fields()
                                .iter()
                                .map(|f| f.name().as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        for row in 0..batch.num_rows() {
            let row_no = sample
                .iter()
                .take(batch_no)
                .map(|b| b.num_rows())
                .sum::<usize>()
                + row
                + 1;
            println!("--- row {row_no} ---");
            for col in &wanted {
                let name = schema.field(*col).name();
                let formatter = &formatters[*col];
                println!("  {name}: {}", cell(&formatter.value(row).to_string()));
            }
        }
    }
    Ok(())
}

/// Collapse the per-column compression codecs of a row group into a short summary.
struct CompressionCounter {
    seen: Vec<(Compression, usize)>,
}

impl FromIterator<Compression> for CompressionCounter {
    fn from_iter<I: IntoIterator<Item = Compression>>(iter: I) -> Self {
        let mut seen = Vec::new();
        for c in iter {
            match seen.iter_mut().find(|(k, _)| *k == c) {
                Some((_, n)) => *n += 1,
                None => seen.push((c, 1)),
            }
        }
        Self { seen }
    }
}

impl CompressionCounter {
    fn describe(&self) -> String {
        if self.seen.is_empty() {
            return "no columns".into();
        }
        self.seen
            .iter()
            .map(|(codec, n)| format!("{n}x {codec:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

const MAX_STATS_CHARS: usize = 80;

fn nulls(n: Option<u64>) -> String {
    n.map_or_else(|| "unknown".to_string(), |n| n.to_string())
}

fn opt_bytes(bytes: Option<&[u8]>) -> String {
    let Some(bytes) = bytes else {
        return "None".to_string();
    };
    let Ok(s) = std::str::from_utf8(bytes) else {
        return format!("{bytes:?}");
    };
    let escaped = escape_control(s);
    let count = escaped.chars().count();
    if count <= MAX_STATS_CHARS {
        format!("\"{escaped}\"")
    } else {
        let cut: String = escaped.chars().take(MAX_STATS_CHARS).collect();
        let rest = count - MAX_STATS_CHARS;
        format!("\"{cut}\"... (+{rest} chars)")
    }
}

fn escape_control(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c => out.push(c),
        }
    }
    out
}

const MAX_CELL_CHARS: usize = 120;

fn cell(value: &str) -> String {
    let escaped = escape_control(value);
    let count = escaped.chars().count();
    if count <= MAX_CELL_CHARS {
        return escaped;
    }
    let cut: String = escaped.chars().take(MAX_CELL_CHARS).collect();
    format!("{cut}... (+{} chars)", count - MAX_CELL_CHARS)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Exit quietly on SIGPIPE (e.g. `courts f.parquet | head`) rather than
    // panicking inside println!.
    sigpipe::reset();
    color_eyre::install().map_err(anyhow::Error::msg)?;
    let cli = Cli::parse();

    tokio::task::spawn_blocking(move || run(&cli))
        .await
        .context("parquet reader task panicked")??;
    Ok(())
}
