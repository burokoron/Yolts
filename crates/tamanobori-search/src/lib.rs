use tamanobori_core::MOVE_LABEL_SIZE;

#[derive(Clone, Debug, PartialEq)]
pub struct SearchModeConfig {
    pub mode_id: &'static str,
    pub alpha_style_self: f32,
    pub alpha_style_opponent_model: f32,
    pub lambda_nyugyoku: f32,
    pub temperature: f32,
}

impl Default for SearchModeConfig {
    fn default() -> Self {
        Self {
            mode_id: "normal",
            alpha_style_self: 0.0,
            alpha_style_opponent_model: 0.0,
            lambda_nyugyoku: 1.0,
            temperature: 0.0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SearchResultError {
    MoveLabelOutOfRange { label: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub selected_move_label: usize,
    pub mode_id: &'static str,
    /// 初期 scaffold では KLENT-0 の結果だけを記録する。
    /// 他の探索実装を追加するときに設定可能にする。
    pub search_type: &'static str,
}

impl SearchResult {
    /// Creates a search result for a selected move label.
    ///
    /// # Errors
    ///
    /// Returns [`SearchResultError::MoveLabelOutOfRange`] when the selected
    /// move label is outside the fixed move label space.
    pub fn new(
        selected_move_label: usize,
        mode_id: &'static str,
    ) -> Result<Self, SearchResultError> {
        if selected_move_label >= MOVE_LABEL_SIZE {
            return Err(SearchResultError::MoveLabelOutOfRange {
                label: selected_move_label,
            });
        }

        Ok(Self {
            selected_move_label,
            mode_id,
            search_type: "klent0",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_search_mode_is_normal_klent0_ready() {
        let config = SearchModeConfig::default();

        assert_eq!(config.mode_id, "normal");
        assert!(config.alpha_style_self.abs() < f32::EPSILON);
        assert!((config.lambda_nyugyoku - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn search_result_accepts_valid_move_label() {
        let result = SearchResult::new(0, "normal").expect("label 0 is valid");

        assert_eq!(result.selected_move_label, 0);
        assert_eq!(result.search_type, "klent0");
    }

    #[test]
    fn search_result_rejects_invalid_move_label() {
        let result = SearchResult::new(MOVE_LABEL_SIZE, "normal");

        assert_eq!(
            result,
            Err(SearchResultError::MoveLabelOutOfRange {
                label: MOVE_LABEL_SIZE,
            })
        );
    }
}
