# Contributing to Stream Motion Animator

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit bug fixes
- ✨ Implement new features
- 🤖 Integrate new AI models
- 🎨 Create example configs
- 🧪 Add tests

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/stream-motion-animator.git
cd stream-motion-animator
```

### 2. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to all public functions/classes
- Keep functions focused and small
- Use meaningful variable names

Example:

```python
def process_frame(
    image: np.ndarray,
    landmarks: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    """
    Process a single frame through the animation pipeline.
    
    Args:
        image: Input image in BGR format
        landmarks: Dictionary containing facial landmarks
        
    Returns:
        Processed frame or None if processing failed
    """
    # Implementation
    pass
```

### Commit Messages

Use clear, descriptive commit messages:

```
✅ Good:
- Add support for custom ONNX models
- Fix memory leak in animation module
- Improve FPS for RTX 3070 users
- Update documentation for Spout setup

❌ Bad:
- Fixed bug
- Update
- Changes
```

### Testing

Add tests for new features:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=src tests/
```

### Documentation

- Update README.md if adding features
- Add docstrings to new functions
- Update relevant docs/ files
- Include examples in docstrings

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Added tests for new features
- [ ] Documentation updated
- [ ] No unnecessary files committed
- [ ] Branch is up to date with main

### 2. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How has this been tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Code follows style guide
```

### 3. Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, it will be merged

## Integrating New AI Models

See [docs/MODEL_INTEGRATION.md](docs/MODEL_INTEGRATION.md) for detailed guide.

Quick checklist:
- [ ] Inherit from `BaseAnimationModel`
- [ ] Implement all abstract methods
- [ ] Add configuration support
- [ ] Include tests
- [ ] Document usage
- [ ] Provide performance benchmarks

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Verify it's not a configuration issue

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what happened

**To Reproduce**
Steps to reproduce:
1. 
2. 
3. 

**Expected behavior**
What should have happened

**System Information**
- OS: [e.g., Windows 11]
- GPU: [e.g., RTX 3080]
- Python: [e.g., 3.10]
- CUDA: [e.g., 11.8]

**Configuration**
```yaml
# Paste relevant config here
```

**Logs**
```
# Paste error message/logs
```

**Additional context**
Any other relevant information
```

## Feature Requests

### Feature Request Template

```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information
```

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

✅ **Positive Behavior:**
- Using welcoming language
- Respecting differing viewpoints
- Accepting constructive criticism
- Focusing on what's best for the community

❌ **Unacceptable Behavior:**
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information
- Other unprofessional conduct

### Enforcement

Report violations to the project maintainers. All reports will be reviewed and investigated.

## Development Setup Details

### Running in Development Mode

```bash
# Run with debug logging
python main.py --image test.jpg --log-level DEBUG

# Enable profiling
python main.py --image test.jpg --enable-metrics

# Test with mock camera
python main.py --image test.jpg --camera 0
```

### Useful Development Commands

```bash
# Check code style
flake8 src/ --max-line-length=100

# Format code
black src/

# Type checking
mypy src/

# Generate documentation
sphinx-build -b html docs/ docs/_build
```

## Project Structure

```
stream-motion-animator/
├── src/              # Source code
│   ├── animation/    # AI models
│   ├── capture/      # Video input
│   ├── tracking/     # Face tracking
│   ├── output/       # Video output
│   ├── pipeline/     # Main pipeline
│   └── utils/        # Utilities
├── tests/            # Unit tests
├── scripts/          # Helper scripts
├── docs/             # Documentation
├── examples/         # Example configs
└── models/           # Model weights (gitignored)
```

## Getting Help

- **Documentation**: Check docs/ folder
- **Discussions**: GitHub Discussions for questions
- **Issues**: GitHub Issues for bugs/features
- **Chat**: [Discord/Slack if available]

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

Thank you for contributing! 🎉
