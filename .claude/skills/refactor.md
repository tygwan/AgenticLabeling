---
name: refactor
description: Refactoring workflow. Improves code structure, reduces duplication, applies patterns. Use for code quality improvements.
---

# Refactoring Skill

## Usage
```
/refactor [target] [--type <refactoring>]
```

### Parameters
- `target`: File, function, or directory
- `--type`: extract | rename | simplify | pattern | all

### Examples
```bash
/refactor src/utils.ts
/refactor src/api/ --type extract
/refactor handleSubmit --type simplify
/refactor src/services/ --type pattern
```

## Refactoring Types

### Extract (`--type extract`)
Extract methods, classes, or modules:

```typescript
// Before: Long function
function processOrder(order) {
  // validation (10 lines)
  // calculation (15 lines)
  // formatting (10 lines)
}

// After: Extracted functions
function processOrder(order) {
  validateOrder(order);
  const total = calculateTotal(order);
  return formatResult(total);
}
```

### Rename (`--type rename`)
Improve naming clarity:

```typescript
// Before
const d = new Date();
const fn = (x) => x * 2;
class Mgr {}

// After
const currentDate = new Date();
const doubleValue = (value) => value * 2;
class OrderManager {}
```

### Simplify (`--type simplify`)
Reduce complexity:

```typescript
// Before: Nested conditions
if (user) {
  if (user.isActive) {
    if (user.hasPermission) {
      return true;
    }
  }
}
return false;

// After: Guard clauses
if (!user) return false;
if (!user.isActive) return false;
if (!user.hasPermission) return false;
return true;
```

### Pattern (`--type pattern`)
Apply design patterns:

```typescript
// Before: Switch statement
function getDiscount(type) {
  switch (type) {
    case 'gold': return 0.2;
    case 'silver': return 0.1;
    default: return 0;
  }
}

// After: Strategy pattern
const discounts = {
  gold: { calculate: () => 0.2 },
  silver: { calculate: () => 0.1 },
  default: { calculate: () => 0 }
};

function getDiscount(type) {
  return (discounts[type] || discounts.default).calculate();
}
```

## Workflow

### Step 1: Analyze Target
```bash
# Find code smells
wc -l {file}  # Long files
Grep: "if.*if.*if|for.*for"  # Deep nesting
Grep: "TODO|FIXME|HACK"  # Technical debt markers
```

### Step 2: Identify Opportunities

| Smell | Indicator | Refactoring |
|-------|-----------|-------------|
| Long Method | >20 lines | Extract Method |
| Large Class | >300 lines | Extract Class |
| Long Params | >3 params | Parameter Object |
| Duplicate Code | Similar blocks | Extract Common |
| Deep Nesting | >3 levels | Guard Clauses |
| God Object | Does everything | Split Responsibilities |

### Step 3: Plan Changes
```markdown
## Refactoring Plan

### Target: src/orderService.ts

### Issues Found
1. `processOrder` is 85 lines (should be <20)
2. Duplicate validation in 3 methods
3. Deep nesting in `calculateDiscount`

### Proposed Changes
1. Extract `validateOrder()`, `calculateTotals()`, `applyDiscounts()`
2. Create `ValidationService` for shared validation
3. Use guard clauses in `calculateDiscount`

### Execution Order
1. Add tests for current behavior
2. Extract methods (preserving behavior)
3. Run tests after each change
4. Commit incrementally
```

### Step 4: Execute Safely
1. Verify tests exist
2. Make one change
3. Run tests
4. Commit
5. Repeat

## Safety Checklist

Before refactoring:
- [ ] Tests exist for target code
- [ ] All tests pass
- [ ] Code is committed

After each change:
- [ ] Tests still pass
- [ ] Behavior unchanged
- [ ] Code compiles

After completion:
- [ ] All tests pass
- [ ] Code review complete
- [ ] Changes committed

## Output Format

```markdown
## Refactoring Report

### Target
`src/services/orderService.ts`

### Changes Made
| Line | Before | After |
|------|--------|-------|
| 45-85 | processOrder (40 lines) | processOrder (10 lines) + 3 helpers |

### New Functions
- `validateOrder(order)` - Validation logic
- `calculateTotals(items)` - Total calculation
- `applyDiscounts(total, customer)` - Discount logic

### Metrics
| Metric | Before | After |
|--------|--------|-------|
| Lines | 85 | 45 |
| Complexity | 12 | 4 |
| Functions | 1 | 4 |

### Tests
All 15 tests passing
```

## Common Refactorings

| Technique | When to Use |
|-----------|-------------|
| Extract Method | Long function |
| Extract Variable | Complex expression |
| Inline Variable | Obvious temporary |
| Extract Class | Large class |
| Move Method | Feature envy |
| Replace Temp with Query | Calculated value |
| Replace Conditional with Polymorphism | Type-based switching |
| Introduce Parameter Object | Many parameters |
