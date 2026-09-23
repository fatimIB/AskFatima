import "../styles/ServiceUnavailableBanner.css";

function ServiceUnavailableBanner({ secondsRemaining }) {
  return (
    <div className="service-unavailable-banner">
      <div className="service-unavailable-title">
        AskFatima is temporarily unavailable
      </div>

      <div className="service-unavailable-message">
        The AI service is experiencing high demand.
        Please try again in a moment.
      </div>

      <div className="service-unavailable-countdown">
        You can try again in{" "}
        <strong>{secondsRemaining}s</strong>
      </div>
    </div>
  );
}

export default ServiceUnavailableBanner;