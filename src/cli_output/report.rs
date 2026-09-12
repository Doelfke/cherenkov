//! Presentation data shared by Markdown and terminal views.

use tabled::{Table, Tabled, builder::Builder, settings::Style};

#[derive(Default)]
pub(super) struct Report {
    pub sections: Vec<Section>,
}

pub(super) enum Section {
    Fields {
        title: String,
        values: Vec<(String, String)>,
    },
    Table {
        title: String,
        data: TableData,
    },
    Note(String),
}

pub(super) struct TableData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Report {
    pub fn fields<T: Tabled>(&mut self, title: &str, value: &T) {
        let values = T::headers()
            .into_iter()
            .zip(value.fields())
            .map(|(name, value)| (name.into_owned(), value.into_owned()))
            .collect();

        self.sections.push(Section::Fields {
            title: title.into(),
            values,
        });
    }

    pub fn table(&mut self, title: &str, table: Table) {
        let mut records: Vec<Vec<String>> = Builder::from(table).into();
        let headers = if records.is_empty() {
            Vec::new()
        } else {
            records.remove(0)
        };

        self.sections.push(Section::Table {
            title: title.into(),
            data: TableData {
                headers,
                rows: records,
            },
        });
    }

    pub fn note(&mut self, text: String) {
        self.sections.push(Section::Note(text));
    }

    pub fn markdown(&self) -> String {
        self.sections
            .iter()
            .map(Section::markdown)
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

impl Section {
    fn markdown(&self) -> String {
        match self {
            Self::Fields { title, values } => {
                let rows = values.iter().map(|(name, value)| [name, value]);
                let mut builder = Builder::from_iter(rows);

                builder.insert_record(0, ["Metric", "Value"]);

                markdown_table(title, builder)
            }
            Self::Table { title, data } => {
                let records = std::iter::once(&data.headers).chain(&data.rows);

                markdown_table(title, Builder::from_iter(records))
            }
            Self::Note(text) => text.clone(),
        }
    }
}

fn markdown_table(title: &str, builder: Builder) -> String {
    let mut table = builder.build();

    table.with(Style::markdown());

    format!("## {title}\n\n{table}")
}
