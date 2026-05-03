"""Readable test output helpers for the MRI QA test suite."""

TEST_DESCRIPTIONS = {
    'test_sanitize_set_name_basic': 'Set name cleanup: spaces become underscores',
    'test_sanitize_set_name_removes_specials': 'Set name cleanup: unsafe characters are removed',
    'test_sanitize_set_name_limits_length': 'Set name cleanup: long names are limited safely',
}

def pytest_report_teststatus(report, config):
    if report.when != 'call':
        return None
    name = report.nodeid.split('::')[-1]
    description = TEST_DESCRIPTIONS.get(name, name.replace('test_', '').replace('_', ' ').title())
    if report.passed:
        return 'passed', f'\nPASS - {description}\n', f'PASS - {description}'
    if report.failed:
        return 'failed', f'\nFAIL - {description}\n', f'FAIL - {description}'
    if report.skipped:
        return 'skipped', f'\nSKIP - {description}\n', f'SKIP - {description}'
    return None
