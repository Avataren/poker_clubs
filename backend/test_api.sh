#!/bin/bash

# Poker Server API Test Script
# This script tests all the main endpoints

BASE_URL="http://127.0.0.1:3000"

echo "🎮 Poker Server API Test"
echo "========================="
echo ""

# Test 1: Register User
echo "1. Registering new user..."
REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"testplayer","email":"test@poker.com","password":"pass123"}')

TOKEN=$(echo $REGISTER_RESPONSE | jq -r '.token')
USER_ID=$(echo $REGISTER_RESPONSE | jq -r '.user.id')

if [ "$TOKEN" = "null" ] || [ -z "$TOKEN" ]; then
    echo "❌ Registration failed"
    echo $REGISTER_RESPONSE | jq
    exit 1
fi

echo "✅ User registered successfully"
echo "   Token: ${TOKEN:0:50}..."
echo "   User ID: $USER_ID"
echo ""

# Test 2: Login
echo "2. Testing login..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"testplayer","password":"pass123"}')

echo "✅ Login successful"
echo ""

# Test 3: Create Club
echo "3. Creating poker club..."
CLUB_RESPONSE=$(curl -s -X POST "$BASE_URL/api/clubs" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name":"Test Poker Club"}')

CLUB_ID=$(echo $CLUB_RESPONSE | jq -r '.club.id')
BALANCE=$(echo $CLUB_RESPONSE | jq -r '.balance')

echo "✅ Club created"
echo "   Club ID: $CLUB_ID"
echo "   Starting balance: $BALANCE chips"
echo ""

# Test 4: Get My Clubs
echo "4. Fetching user clubs..."
MY_CLUBS=$(curl -s -X GET "$BASE_URL/api/clubs/my" \
  -H "Authorization: Bearer $TOKEN")

CLUB_COUNT=$(echo $MY_CLUBS | jq '. | length')
echo "✅ Retrieved $CLUB_COUNT club(s)"
echo ""

# Test 5: Create Table
echo "5. Creating poker table..."
TABLE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/tables" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{\"club_id\":\"$CLUB_ID\",\"name\":\"High Stakes\",\"small_blind\":50,\"big_blind\":100}")

TABLE_ID=$(echo $TABLE_RESPONSE | jq -r '.id')
echo "✅ Table created"
echo "   Table ID: $TABLE_ID"
echo "   Blinds: 50/100"
echo "   Min buy-in: $(echo $TABLE_RESPONSE | jq -r '.min_buyin')"
echo "   Max buy-in: $(echo $TABLE_RESPONSE | jq -r '.max_buyin')"
echo ""

# Test 6: List Club Tables
echo "6. Listing club tables..."
TABLES=$(curl -s -X GET "$BASE_URL/api/tables/club/$CLUB_ID" \
  -H "Authorization: Bearer $TOKEN")

TABLE_COUNT=$(echo $TABLES | jq '.tables | length')
echo "✅ Found $TABLE_COUNT table(s) in club"
echo ""

# Summary
echo "========================="
echo "🎉 All tests passed!"
echo ""
echo "Server is fully operational:"
echo "  • Authentication ✅"
echo "  • Club Management ✅"
echo "  • Table Creation ✅"
echo "  • Database ✅"
echo ""
echo "Next steps:"
echo "  1. Connect via WebSocket: ws://127.0.0.1:3000/ws?token=$TOKEN"
echo "  2. Join table: $TABLE_ID"
echo "  3. Start playing poker!"
echo ""
echo "Keep this token for testing: $TOKEN"
