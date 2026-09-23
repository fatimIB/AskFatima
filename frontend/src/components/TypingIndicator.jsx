import "../styles/BotMessage.css"
import "../styles/Typing.css"
import profileImage from "../assets/Profile.png";


function TypingIndicator() {
    return (
        <div className="bot-message">

            <div className="bot-avatar">
                <img
                    src={profileImage}
                    alt="profile"
                />
            </div>

            <div className="bot-bubble">

                <div className="typing">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>

            </div>

        </div>
    );
}

export default TypingIndicator;