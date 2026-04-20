---
name: test
description: Testing workflow. Generates tests, analyzes coverage, suggests test cases. Use for improving test coverage or creating new tests.
---

# Testing Skill

## Usage
```
/test [action] [target]
```

### Parameters
- `action`: generate | coverage | run | suggest
- `target`: File, function, or directory

### Examples
```bash
/test generate src/utils.ts
/test coverage
/test run src/auth.test.ts
/test suggest src/api/users.ts
```

## Actions

### Generate Tests (`/test generate`)
Creates test file for target:

```bash
/test generate src/calculator.ts
```

Output: `src/calculator.test.ts` or `tests/calculator.test.ts`

### Coverage Analysis (`/test coverage`)
Analyzes and reports test coverage:

```markdown
## Coverage Report

| File | Lines | Functions | Branches |
|------|-------|-----------|----------|
| src/auth.ts | 85% | 90% | 75% |
| src/utils.ts | 95% | 100% | 90% |
| **Total** | **88%** | **92%** | **80%** |

### Uncovered Lines
- src/auth.ts:45-52 (error handling)
- src/api.ts:120-125 (edge case)
```

### Run Tests (`/test run`)
Executes tests with output:

```bash
# Detect framework and run
npm test
pytest
dotnet test
```

### Suggest Tests (`/test suggest`)
Analyzes code and suggests test cases:

```markdown
## Suggested Test Cases for `src/auth.ts`

### `login(email, password)`
- [ ] Valid credentials → returns token
- [ ] Invalid email → throws AuthError
- [ ] Invalid password → throws AuthError
- [ ] Empty email → throws ValidationError
- [ ] Empty password → throws ValidationError
- [ ] Locked account → throws AccountLocked
- [ ] Rate limited → throws RateLimitError

### `logout(token)`
- [ ] Valid token → returns true
- [ ] Invalid token → throws InvalidToken
- [ ] Expired token → throws TokenExpired
```

## Test Templates

### JavaScript/TypeScript (Jest)
```typescript
import { functionName } from './module';

describe('functionName', () => {
  describe('happy path', () => {
    it('should return expected result', () => {
      expect(functionName(input)).toBe(expected);
    });
  });

  describe('edge cases', () => {
    it('should handle empty input', () => {
      expect(functionName('')).toBe(default);
    });

    it('should handle null', () => {
      expect(functionName(null)).toBeNull();
    });
  });

  describe('error cases', () => {
    it('should throw on invalid input', () => {
      expect(() => functionName(invalid)).toThrow(Error);
    });
  });
});
```

### Python (pytest)
```python
import pytest
from module import function_name

class TestFunctionName:
    """Tests for function_name."""

    def test_happy_path(self):
        """Should return expected result."""
        assert function_name(input) == expected

    def test_empty_input(self):
        """Should handle empty input."""
        assert function_name('') == default

    def test_invalid_input(self):
        """Should raise on invalid input."""
        with pytest.raises(ValueError):
            function_name(invalid)

    @pytest.mark.parametrize('input,expected', [
        ('a', 1),
        ('b', 2),
        ('c', 3),
    ])
    def test_multiple_cases(self, input, expected):
        """Should handle multiple cases."""
        assert function_name(input) == expected
```

### C# (xUnit)
```csharp
public class FunctionNameTests
{
    [Fact]
    public void Should_ReturnExpected_WhenValidInput()
    {
        // Arrange
        var sut = new MyClass();

        // Act
        var result = sut.FunctionName(input);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("a", 1)]
    [InlineData("b", 2)]
    public void Should_Handle_MultipleCases(string input, int expected)
    {
        var sut = new MyClass();
        Assert.Equal(expected, sut.FunctionName(input));
    }

    [Fact]
    public void Should_Throw_WhenInvalidInput()
    {
        var sut = new MyClass();
        Assert.Throws<ArgumentException>(() => sut.FunctionName(invalid));
    }
}
```

## Workflow

### 1. Detect Test Framework
```bash
Grep: "jest|vitest|mocha" package.json
Grep: "pytest|unittest" requirements.txt
Grep: "xunit|nunit" *.csproj
```

### 2. Analyze Target Code
- Function signatures
- Input types
- Return types
- Error conditions
- Dependencies (for mocking)

### 3. Generate Test Cases
- Happy path (normal operation)
- Edge cases (boundaries, empty, null)
- Error cases (invalid input, failures)
- Integration scenarios

### 4. Create Test File
- Follow project conventions
- Use appropriate assertions
- Add setup/teardown if needed
- Include mocks for dependencies

## Best Practices

1. **Test behavior, not implementation**
2. **One assertion per test** (when practical)
3. **Descriptive test names**
4. **Arrange-Act-Assert pattern**
5. **Mock external dependencies**
6. **Test edge cases first**
