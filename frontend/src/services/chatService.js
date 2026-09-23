const API_URL = import.meta.env.VITE_API_URL || "";

/**
 * Sends a question with conversation history to the backend.
 *
 * Returns:
 * - Normal response: { answer, sources }
 * - Rate-limited response: { rate_limited: true }
 * - Service unavailable response: { service_unavailable: true }
 */
export async function sendMessage(question, chatHistory = []) {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      chat_history: chatHistory,
    }),
  });

  const data = await response.json();
  
  console.log("Chat response:", data);

  // Backend rate-limit response
  if (
    response.status === 429 ||
    data.rate_limited === true ||
    data.detail?.rate_limited === true
  ) {
    return {
      rate_limited: true,
    };
  }

  // AI provider temporarily unavailable
  if (response.status === 503) {
    return {
      service_unavailable: true,
      answer: null,
    };
  }

  // Other backend errors
  if (!response.ok) {
    throw new Error(
      "Something went wrong reaching AskFatima. Please try again.",
    );
  }

  // Normal successful response
  return {
    answer: data.answer,
    sources: data.sources,
  };
}
