import "../styles/BotMessage.css";
import profileImage from "../assets/Profile.png";
import { useState } from "react";
import ReactMarkdown from "react-markdown";

function BotMessage({ message, sources = [] }) {
    const [showSources, setShowSources] = useState(false);

    return (
        <div className="bot-message">

            <div className="bot-avatar">
                <img
                    src={profileImage}
                    alt="profile"
                />
            </div>

            <div className="bot-bubble">

                <div className="bot-answer">
                    <ReactMarkdown>
                        {message}
                    </ReactMarkdown>
                </div>

                {sources.length > 0 && (
                    <div className="bot-sources">

                        <button
                            className="bot-sources__toggle"
                            onClick={() => setShowSources((prev) => !prev)}
                        >
                            <span>
                                {showSources ? "Hide sources" : "View sources"}
                            </span>

                            <span className="bot-sources__count">
                                {sources.length}
                            </span>
                        </button>

                        {showSources && (
                            <div className="bot-sources__list">

                                {sources.map((source, index) => (
                                    <div
                                        className="bot-source"
                                        key={index}
                                    >
                                        <span className="bot-source__file">
                                            {source.source}
                                        </span>

                                        <span className="bot-source__separator">
                                            ·
                                        </span>

                                        <span className="bot-source__section">
                                            {source.section}
                                        </span>
                                    </div>
                                ))}

                            </div>
                        )}

                    </div>
                )}

            </div>

        </div>
    );
}

export default BotMessage;