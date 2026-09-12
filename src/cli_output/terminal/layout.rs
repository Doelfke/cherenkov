//! Measure report sections and build native Ratatui widgets.

use super::{Report, Theme};
use crate::cli_output::{Section, TableData};
use anyhow::{Context, Result};
use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Rect},
    style::Style,
    text::{Line, Text},
    widgets::{Cell, Paragraph, Row, Table, Widget, Wrap},
};

pub(in crate::cli_output) struct Panel {
    pub height: u16,
    body: Body,
}

enum Body {
    Text(Paragraph<'static>),
    Table(Table<'static>),
}

impl Panel {
    pub fn render(self, area: Rect, buffer: &mut Buffer) {
        match self.body {
            Body::Text(text) => text.render(area, buffer),
            Body::Table(table) => table.render(area, buffer),
        }
    }
}

pub(in crate::cli_output) fn panels(
    report: &Report,
    width: u16,
    theme: Theme,
) -> Result<Vec<Panel>> {
    let mut panels = Vec::new();

    for section in &report.sections {
        match section {
            Section::Fields { title, values } => {
                panels.push(paragraph(title, width, theme.heading)?);
                panels.push(fields(values, width, theme)?);
            }
            Section::Table { title, data } => {
                panels.push(paragraph(title, width, theme.heading)?);
                panels.extend(table(data, width, theme)?);
            }
            Section::Note(text) => panels.push(paragraph(text, width, theme.label)?),
        }

        panels.push(paragraph("", width, Style::default())?);
    }

    Ok(panels)
}

fn paragraph(text: &str, width: u16, style: Style) -> Result<Panel> {
    let text = Paragraph::new(text.to_owned())
        .style(style)
        .wrap(Wrap { trim: false });
    let height = height(text.line_count(width))?;

    Ok(Panel {
        height,
        body: Body::Text(text),
    })
}

fn fields(values: &[(String, String)], width: u16, theme: Theme) -> Result<Panel> {
    if width < 8 {
        let text = values
            .iter()
            .map(|(name, value)| format!("{name}\n{value}"))
            .collect::<Vec<_>>()
            .join("\n");

        return paragraph(&text, width, Style::default());
    }

    let value = values
        .iter()
        .map(|(_, value)| display_width(value))
        .max()
        .unwrap_or(1)
        .clamp(1, usize::from(width) * 2 / 3);
    let label = values
        .iter()
        .map(|(name, _)| display_width(name))
        .max()
        .unwrap_or(1)
        .clamp(1, usize::from(width).saturating_sub(value + 2));
    let widths = [label, value];
    let mut rows = Vec::new();
    let mut total = 0;

    for (name, value) in values {
        let (row, height) = wrapped_row(
            &[name.clone(), value.clone()],
            &widths,
            &[false, numeric(value)],
            theme.label,
        )?;

        rows.push(row);

        total += height;
    }

    let table = Table::new(rows, widths.map(|w| Constraint::Length(w as u16))).column_spacing(2);

    Ok(Panel {
        height: height(total)?,
        body: Body::Table(table),
    })
}

fn table(data: &TableData, width: u16, theme: Theme) -> Result<Vec<Panel>> {
    if data.rows.is_empty() {
        return Ok(vec![paragraph("No entries.", width, theme.label)?]);
    }

    let widths: Vec<usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(i, name)| {
            data.rows
                .iter()
                .filter_map(|row| row.get(i))
                .map(|value| display_width(value))
                .chain(std::iter::once(display_width(name)))
                .max()
                .unwrap_or(1)
        })
        .collect();
    let natural = widths.iter().sum::<usize>() + widths.len().saturating_sub(1) * 2;

    if natural > usize::from(width) {
        return cards(data, width, theme);
    }

    let numeric: Vec<bool> = (0..data.headers.len())
        .map(|i| {
            data.rows
                .iter()
                .all(|row| row.get(i).is_some_and(|value| numeric(value)))
        })
        .collect();
    let mut rows = Vec::new();
    let mut total = 0;
    let (header, _) = wrapped_row(&data.headers, &widths, &numeric, Style::default())?;

    rows.push(header.style(theme.header));

    let rules: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();

    rows.push(Row::new(rules).style(theme.rule));

    for values in &data.rows {
        let (row, height) = wrapped_row(values, &widths, &numeric, Style::default())?;

        rows.push(row);

        total += height;
    }

    let constraints = widths.iter().map(|w| Constraint::Length(*w as u16));
    let table = Table::new(rows, constraints).column_spacing(2);

    Ok(vec![Panel {
        height: height(total + 2)?,
        body: Body::Table(table),
    }])
}

fn cards(data: &TableData, width: u16, theme: Theme) -> Result<Vec<Panel>> {
    let mut panels = Vec::new();

    for values in &data.rows {
        let fields_data: Vec<_> = data
            .headers
            .iter()
            .cloned()
            .zip(values.iter().cloned())
            .collect();

        panels.push(fields(&fields_data, width, theme)?);
        panels.push(paragraph("", width, Style::default())?);
    }

    Ok(panels)
}

fn wrapped_row(
    values: &[String],
    widths: &[usize],
    numeric: &[bool],
    first_style: Style,
) -> Result<(Row<'static>, usize)> {
    let mut cells = Vec::new();
    let mut height = 1;

    for (index, ((value, width), numeric)) in values.iter().zip(widths).zip(numeric).enumerate() {
        let lines: Vec<_> = textwrap::wrap(value, (*width).max(1))
            .into_iter()
            .map(|line| Line::raw(line.into_owned()))
            .collect();
        height = height.max(lines.len());
        let alignment = if *numeric {
            Alignment::Right
        } else {
            Alignment::Left
        };
        let style = if index == 0 {
            first_style
        } else {
            Style::default()
        };

        cells.push(Cell::from(Text::from(lines).alignment(alignment)).style(style));
    }

    Ok((Row::new(cells).height(self::height(height)?), height))
}

fn display_width(value: &str) -> usize {
    Line::raw(value).width()
}

// Parsing chooses alignment only; the original text retains full precision.
fn numeric(value: &str) -> bool {
    value == "n/a" || value.parse::<f64>().is_ok()
}

fn height(lines: usize) -> Result<u16> {
    u16::try_from(lines).context("report section is too tall for terminal rendering")
}
