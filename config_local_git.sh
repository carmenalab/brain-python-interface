#!/bin/bash
## Mark certain "config" files as files which git should not check for local changes
## These files are needed to run experiments so they must be in the repository but each local rig may have it's own configuration
## skip-worktree vs assume-unchanged
## http://stackoverflow.com/questions/13630849/git-difference-between-assume-unchanged-and-skip-worktree
git update-index --skip-worktree db/settings.py
