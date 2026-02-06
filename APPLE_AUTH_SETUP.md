# Apple Login Setup

This project supports Sign in with Apple by validating Apple ID tokens on the backend. The backend expects credentials to be present as environment variables (see `.env` for placeholders).

## Required environment variables

```
APPLE_CLIENT_ID=com.example.pokerclubs
APPLE_TEAM_ID=TEAMID12345
APPLE_KEY_ID=KEYID12345
APPLE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----
```

## Apple Developer configuration

1. Enroll in the Apple Developer Program: https://developer.apple.com/
2. Go to **Certificates, Identifiers & Profiles**.
3. Create or select an **App ID** (this is your `APPLE_CLIENT_ID` for native apps).
4. Enable **Sign in with Apple** capability on the App ID.
5. Create a **Services ID** (used as the `APPLE_CLIENT_ID` for web/other flows).
6. Create a **Sign in with Apple Key** and download the `.p8` file.
7. Record the **Key ID** (`APPLE_KEY_ID`) and **Team ID** (`APPLE_TEAM_ID`).
8. Paste the `.p8` private key content into `APPLE_PRIVATE_KEY`.

## Firebase Authentication option

Firebase Authentication can broker Apple sign-in for client apps, but you still need Apple Developer configuration to enable the provider.

1. In Firebase, enable **Apple** under **Authentication → Sign-in method**.
2. Provide the Service ID, Team ID, and Key ID, and upload the private key.
3. Use Firebase SDKs in the client to obtain an ID token.
4. The backend can validate the Apple ID token directly (current implementation) or accept the Firebase ID token if you extend verification to Firebase.

## Notes

- Apple tokens are validated against Apple’s JWKS endpoint.
- Apple may provide relay emails; treat them as valid if `email_verified` is true.
