use super::*;

#[test]
fn bounded_header_reader_rejects_truncation_and_long_lines() {
    let mut n = 4;

    assert!(line(&mut std::io::Cursor::new(b"abcdef\n"), &mut n).is_err());

    let mut n = 16;

    assert!(line(&mut std::io::Cursor::new(b"missing newline"), &mut n).is_err());

    let mut n = 16;

    assert_eq!(
        line(&mut std::io::Cursor::new(b"ok\r\n"), &mut n).unwrap(),
        "ok"
    );
}
