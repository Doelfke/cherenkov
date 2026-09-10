use super::*;

#[test]
fn matching_checks_mtp_lookahead_and_exact_prompt_state() {
    assert!(matches(&[1, 2], Some(3), false, &[1, 2, 3, 4]));
    assert!(!matches(&[1, 2], Some(3), false, &[1, 2, 9]));
    assert!(!matches(&[1, 2], Some(3), false, &[1, 2]));
    assert!(matches(&[1, 2], Some(3), true, &[1, 2]));
    assert!(matches(&[1, 2], None, false, &[1, 2, 9]));
    assert!(!matches(&[1, 2], None, true, &[1, 9, 2]));
    assert!(!matches(&[1, 2], None, true, &[1]));
}
