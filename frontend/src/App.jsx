import "./App.css";

import ChatWindow from "./components/ChatWindow";
import Sidebar from "./components/Sidebar";
import { useEffect, useState } from "react";
import { sendMessage } from "./services/chatService";

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [rateLimited, setRateLimited] = useState(false);

  // Temporary AI service unavailable state
  const [serviceUnavailable, setServiceUnavailable] = useState(false);
  const [secondsRemaining, setSecondsRemaining] = useState(0);

  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hi! I'm AskFatima.",
      sources: [],
    },
  ]);

  // Conversation history in the {role, content} shape the backend expects
  const [chatHistory, setChatHistory] = useState([]);

  // 60-second temporary cooldown after a 503
  useEffect(() => {
    if (!serviceUnavailable) {
      return;
    }

    if (secondsRemaining <= 0) {
      setServiceUnavailable(false);
      return;
    }

    const timer = setInterval(() => {
      setSecondsRemaining((seconds) => seconds - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [serviceUnavailable, secondsRemaining]);

  async function handleSend(text) {
    if (loading || rateLimited || serviceUnavailable) {
      return;
    }

    setError(null);
    setLoading(true);

    // Show user's message immediately
    const userMessage = {
      sender: "user",
      text,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
    ]);

    try {
      const result = await sendMessage(
        text,
        chatHistory
      );

      // Gemini quota/rate limit reached
      if (result.rate_limited) {
        setRateLimited(true);

        // Remove the user's message because
        // the request did not receive an answer.
        setMessages((previous) =>
          previous.slice(0, -1)
        );

        return false;
      }

      // Gemini temporarily unavailable
      if (result.service_unavailable) {
        setServiceUnavailable(true);
        setSecondsRemaining(60);

        // Remove the user's message because
        // the request did not receive an answer.
        setMessages((previous) =>
          previous.slice(0, -1)
        );

        return false;
      }

      // Normal response
      const botMessage = {
        sender: "bot",
        text: result.answer,
        sources: result.sources,
      };

      setMessages((previous) => [
        ...previous,
        botMessage,
      ]);

      // Update conversation history
      setChatHistory((previous) => [
        ...previous,
        {
          role: "user",
          content: text,
        },
        {
          role: "assistant",
          content: result.answer,
        },
      ]);

      return true;

    } catch (err) {
      setError(
        err.message ||
        "Something went wrong. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <Sidebar />

      <div className="chat-area">
        <ChatWindow
          messages={messages}
          loading={loading}
          error={error}
          rateLimited={rateLimited}
          serviceUnavailable={serviceUnavailable}
          secondsRemaining={secondsRemaining}
          onSend={handleSend}
        />
      </div>
    </div>
  );
}

export default App;