# Responsive Layout Fixes

## Summary
Fixed the poker client window resizing issues where cards, game info, and table elements were moving differently. All elements now scale coherently across different aspect ratios.

## Changes Made

### 1. game_screen.dart
**Problem**: Community cards and tournament info used fixed positioning (`top: 180`) while the table used responsive `LayoutBuilder` calculations.

**Solution**: 
- Wrapped the entire game area in a `LayoutBuilder` to get consistent dimensions
- Made community cards position responsive relative to table size
- Cards now positioned at `tableHeight * 0.4` from center (40% down)
- Card sizes scale based on table dimensions:
  - Width: `(tableWidth * 0.08).clamp(40, 60)`
  - Height: `(tableHeight * 0.13).clamp(56, 84)`
- Tournament info inherits same responsive positioning

### 2. table_seat_widget.dart - PokerTableWidget
**Problem**: Fixed sizes and positions caused elements to misalign during window resize.

**Solution**:
- **Table Border**: Now responsive `(tableWidth * 0.012).clamp(4.0, 10.0)`
- **Pot Position**: Changed from fixed offset `-125` to responsive `tableHeight * 0.25` (25% from center)
- **Chip Stack Scaling**: Dynamic scaling based on table width `(tableWidth / 500).clamp(0.8, 1.3)`
- **Seat Positioning**: 
  - Radius calculations now use percentages: `tableWidth * 0.05` instead of fixed `30`
  - Responsive seat sizes: `(tableWidth * 0.13).clamp(60.0, 90.0)`
  - Card dimensions scale with table

### 3. table_seat_widget.dart - TableSeatWidget
**Problem**: Fixed font sizes and dimensions looked bad at different window sizes.

**Solution**: Made ALL dimensions responsive:
- **Seat Circle**: Dynamic size passed from parent `seatSize = (tableWidth * 0.13).clamp(60, 90)`
- **Card Sizes**: 
  - Width: `(tableWidth * 0.058).clamp(28, 42)`
  - Height: `(tableHeight * 0.095).clamp(40, 60)`
- **Font Sizes**: All calculated as percentages of seat size
  - Username: `(seatSize * 0.14).clamp(9, 13)`
  - Stack: `(seatSize * 0.15).clamp(10, 14)`
  - Bet: `(seatSize * 0.11).clamp(8, 11)`
  - Badges: `(seatSize * 0.11).clamp(8, 11)`
- **Spacing**: All paddings/margins use multipliers of `seatSize`
- **Badge Positions**: Relative to seat size (`seatSize * 0.05`)
- **Icon Sizes**: Scaled to `seatSize * 0.15`

### 4. Chip Stack and Card Dealing Animations
**Problem**: Fixed positions for chips and dealing animations.

**Solution**:
- Chip stacks positioned using same responsive radius calculations
- Chip stack scale: `(tableWidth / 600).clamp(0.7, 1.2)`
- Card dealing animation uses responsive card dimensions
- Card offset for second card: `cardWidth * 0.57` (instead of fixed `20.0`)

## Benefits

### Current Benefits
1. **Coherent Scaling**: All elements maintain proper relative positions during resize
2. **Aspect Ratio Support**: Works correctly across different window aspect ratios
3. **No Layout Shifts**: Elements stay visually aligned with the table
4. **Smooth Animations**: Card dealing and pot animations work at any size

### Future Mobile Support
The responsive design is already prepared for mobile:
- Uses `.clamp()` to set minimum and maximum sizes
- Percentage-based positioning works on any screen
- Touch targets (seats, buttons) will scale appropriately
- Text remains readable at small sizes (9px minimum)

## Testing Recommendations

1. **Desktop Testing**:
   - Resize window from very narrow to very wide
   - Test different aspect ratios (16:9, 4:3, ultrawide)
   - Verify all elements stay aligned with table

2. **Future Mobile Testing**:
   - Portrait mode (will need UI adjustments for buttons)
   - Landscape mode (should work well with current implementation)
   - Different device sizes (phone vs tablet)

## Notes for Mobile View (Future Work)

When implementing mobile view, consider:
1. **Portrait Mode**: May need to stack action buttons vertically or use a bottom sheet
2. **Touch Targets**: Current seat size minimum of 60px should be fine, but consider increasing to 72px for better touch
3. **Font Sizes**: Current minimums (9-13px) are reasonable but test readability
4. **Tournament Info**: May need to be collapsed or moved for small screens
5. **Community Cards**: Might need to scale smaller in portrait mode

## Files Modified
- `/home/avataren/src/poker/poker_client/lib/screens/game_screen.dart`
- `/home/avataren/src/poker/poker_client/lib/widgets/table_seat_widget.dart`
