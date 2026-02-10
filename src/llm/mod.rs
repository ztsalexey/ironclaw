//! LLM integration for the agent.
//!
//! Supports two API modes:
//! - **Responses API** (chat-api): Session-based auth, uses `/v1/responses` endpoint
//! - **Chat Completions API** (cloud-api): API key auth, uses `/v1/chat/completions` endpoint

pub mod failover;
mod nearai;
mod nearai_chat;
mod provider;
mod reasoning;
mod retry;
pub mod session;

pub use failover::FailoverProvider;
pub use nearai::{ModelInfo, NearAiProvider};
pub use nearai_chat::NearAiChatProvider;
pub use provider::{
    ChatMessage, CompletionRequest, CompletionResponse, LlmProvider, Role, ToolCall,
    ToolCompletionRequest, ToolCompletionResponse, ToolDefinition, ToolResult,
};
pub use reasoning::{ActionPlan, Reasoning, ReasoningContext, RespondResult, ToolSelection};
pub use session::{SessionConfig, SessionManager, create_session_manager};

use std::sync::Arc;

use crate::config::{LlmConfig, NearAiApiMode, NearAiConfig};
use crate::error::LlmError;

/// Create an LLM provider based on configuration.
///
/// - For `Responses` mode: Requires a session manager for authentication
/// - For `ChatCompletions` mode: Uses API key from config (session not needed)
pub fn create_llm_provider(
    config: &LlmConfig,
    session: Arc<SessionManager>,
) -> Result<Arc<dyn LlmProvider>, LlmError> {
    create_llm_provider_with_config(&config.nearai, session)
}

/// Create an LLM provider from a `NearAiConfig` directly.
///
/// This is useful when constructing additional providers for failover,
/// where only the model name differs from the primary config.
pub fn create_llm_provider_with_config(
    config: &NearAiConfig,
    session: Arc<SessionManager>,
) -> Result<Arc<dyn LlmProvider>, LlmError> {
    match config.api_mode {
        NearAiApiMode::Responses => {
            tracing::info!(
                model = %config.model,
                "Using Responses API (chat-api) with session auth"
            );
            Ok(Arc::new(NearAiProvider::new(config.clone(), session)))
        }
        NearAiApiMode::ChatCompletions => {
            tracing::info!(
                model = %config.model,
                "Using Chat Completions API (cloud-api) with API key auth"
            );
            Ok(Arc::new(NearAiChatProvider::new(config.clone())?))
        }
    }
}
