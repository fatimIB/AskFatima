import "../styles/ChatWindow.css";

import UserMessage from "./UserMessage";
import BotMessage from "./BotMessage";
import TypingIndicator from "./TypingIndicator";
import ErrorMessage from "./ErrorMessage";
import ChatInput from "./ChatInput";
import RateLimitBanner from "./RateLimitBanner";
import ServiceUnavailableBanner from "./ServiceUnavailableBanner";

function ChatWindow({
    messages,
    loading,
    error,
    rateLimited,
    serviceUnavailable,
    secondsRemaining,
    onSend,
}) {
    return (
        <main className="chat-window">

            <div className="messages">

                {messages.map((message, index) =>
                    message.sender === "user" ? (
                        <UserMessage
                            key={index}
                            message={message.text}
                        />
                    ) : (
                        <BotMessage
                            key={index}
                            message={message.text}
                            sources={message.sources}
                        />
                    )
                )}

                {loading && <TypingIndicator />}

                {error && (
                    <ErrorMessage message={error} />
                )}

            </div>

            {/* Rate-limit message */}
            {rateLimited && (
                <RateLimitBanner />
            )}

            {/* Temporary AI service unavailable message */}
            {serviceUnavailable && (
                <ServiceUnavailableBanner
                    secondsRemaining={secondsRemaining}
                />
            )}

            <ChatInput
                onSend={onSend}
                disabled={
                    loading ||
                    rateLimited ||
                    serviceUnavailable
                }
            />

        </main>
    );
}

export default ChatWindow;