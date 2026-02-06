# Google Login Setup

This project supports Google sign-in by validating Google ID tokens on the backend. The backend expects credentials to be present as environment variables (see `.env` for placeholders).

## Required environment variables

```
GOOGLE_CLIENT_ID=google-client-id.apps.googleusercontent.com
```

## Option A: Google Cloud OAuth Client (direct)

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project.
3. Go to **APIs & Services → OAuth consent screen** and configure the consent screen.
4. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**.
5. Choose the appropriate application type (Web, iOS, Android).
6. Copy the **Client ID** into `GOOGLE_CLIENT_ID`.

## Option B: Firebase Authentication (recommended for mobile/web)

Firebase can be used to manage Google sign-in for all clients (web, iOS, Android) while the backend validates Firebase-issued ID tokens.

1. Create or select a Firebase project: https://console.firebase.google.com/
2. Navigate to **Authentication → Sign-in method** and enable **Google**.
3. For mobile/web, configure platform-specific setup as prompted by Firebase.
4. In your client app, use the Firebase SDK to obtain an ID token.
5. Set `GOOGLE_CLIENT_ID` to the Firebase project’s OAuth client ID (listed under project settings → General).

## Notes

- The backend validates JWT signatures against Google’s public JWKS endpoint.
- Make sure your client app requests an ID token for the same client ID configured in `GOOGLE_CLIENT_ID`.
