import os

from setuptools import find_packages, setup

REPO_DIR = os.path.dirname(os.path.realpath(__file__))


def getVersion():
    """
    Get version from local file.
    """
    with open(os.path.join(REPO_DIR, "VERSION"), "r") as versionFile:
        return versionFile.read().strip()


def parse_file(requirementFile):
    try:
        return [
            line.strip()
            for line in open(requirementFile).readlines()
            if not line.startswith("-")
        ]
    except IOError:
        return []


def findRequirements():
    """
    Read the requirements.txt file and parse into requirements for setup's
    install_requirements option.
    """
    requirementsPath = os.path.join(REPO_DIR, "requirements.txt")
    return parse_file(requirementsPath)


if __name__ == "__main__":
    requirements = findRequirements()
    REPO_NAME = "Uber_Project_Pipeline"
    AUTHOR_USER_NAME = "MLFOLKS"
    SRC_REPO = "UberModel"
    setup(
        name=SRC_REPO,
        version=getVersion(),
        description='Uber End To End Pipeline',
        author=AUTHOR_USER_NAME,
        author_email='thanseefpp@gmail.com',
        url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
        project_urls={
            "Bug tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        },
        packages=find_packages(),
        install_requires=requirements,
    )
