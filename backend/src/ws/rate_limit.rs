use std::time::Instant;

/// Token bucket rate limiter for WebSocket connections.
///
/// Allows `rate` messages per second with a burst capacity of `burst`.
pub struct RateLimiter {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// * `rate` - messages per second (steady state)
    /// * `burst` - maximum burst capacity
    pub fn new(rate: f64, burst: f64) -> Self {
        Self {
            tokens: burst,
            max_tokens: burst,
            refill_rate: rate,
            last_refill: Instant::now(),
        }
    }

    /// Check if a message is allowed. Returns true if allowed, false if rate limited.
    pub fn allow(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_burst_allowed() {
        let mut rl = RateLimiter::new(10.0, 20.0);
        // Should allow up to burst limit
        for _ in 0..20 {
            assert!(rl.allow());
        }
        // Next one should be denied
        assert!(!rl.allow());
    }

    #[test]
    fn test_refill() {
        let mut rl = RateLimiter::new(10.0, 5.0);
        // Use up all tokens
        for _ in 0..5 {
            assert!(rl.allow());
        }
        assert!(!rl.allow());

        // Wait for refill (100ms at 10/sec = 1 token)
        thread::sleep(Duration::from_millis(120));
        assert!(rl.allow());
    }

    #[test]
    fn test_does_not_exceed_max() {
        let mut rl = RateLimiter::new(10.0, 5.0);
        // Wait a long time
        thread::sleep(Duration::from_millis(200));
        // Should still only have max_tokens
        let mut count = 0;
        while rl.allow() {
            count += 1;
            if count > 100 {
                break; // Safety valve
            }
        }
        assert_eq!(count, 5);
    }
}
