from django.test.utils import get_runner
from django.conf import settings
from bmi3d.boot_django import boot_django

boot_django()
TestRunner = get_runner(settings)
test_runner = TestRunner()
failures = test_runner.run_tests(["tests/django"])
