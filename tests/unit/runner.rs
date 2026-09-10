use super::*;

#[test]
fn context_budget_includes_speculative_lookahead() {
    assert!(check_budget(10, 5, 2, 17).is_ok());
    assert!(check_budget(10, 5, 2, 16).is_err());
    assert!(check_budget(17, 1, 0, 16).is_err());
    assert!(check_budget(0, 5, 2, 17).is_err());
    assert!(check_budget(usize::MAX, 1, 0, usize::MAX).is_err());
}
