import "../styles/ChatInput.css"
import { useState } from "react";
import { IoSend } from "react-icons/io5";

function ChatInput({ onSend, disabled = false }) {
  const [message, setMessage] = useState("");

  async function handleSend() {

    if (disabled) return;

    const trimmedMessage = message.trim();

    if (!trimmedMessage) return;

    onSend(trimmedMessage); 
    setMessage("");
  }

  function handleKeyDown(event) {
    // Send on Enter
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-input-container">
      <textarea
        className="chat-input"
        placeholder="Ask me anything about Fatima..."
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={1}
      />

      <button
        className="send-button"
        onClick={handleSend}
        disabled={disabled || !message.trim()}
      >
        <IoSend size={16} />
      </button>
    </div>
  );
}

export default ChatInput;