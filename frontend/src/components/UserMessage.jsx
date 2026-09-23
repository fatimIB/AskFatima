import "../styles/UserMessage.css"



function UserMessage({ message }) {
    return (
        <div className="user-message">
            <div className="user-message__bubble">
                {message}
            </div>
        </div>
    );
}

export default UserMessage;