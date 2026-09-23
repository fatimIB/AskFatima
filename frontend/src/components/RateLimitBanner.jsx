import "../styles/RateLimitBanner.css";

function RateLimitBanner() {
  return (
    <div className="rate-limit-banner">
      <div className="rate-limit-icon">✦</div>

      <div className="rate-limit-content">
        <div className="rate-limit-title">
          You've reached today's usage limit
        </div>

        <div className="rate-limit-message">
          Please try again tomorrow.
        </div>
      </div>
    </div>
  );
}

export default RateLimitBanner;

