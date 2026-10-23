# Contributing

## Workflow

1. Branch from `main` (a `git worktree` keeps `main` clean while you work).
2. For any change beyond a typo, write `docs/plan/YYYY-MM-DD-<topic>.md`: what was wrong, what changed, what was deliberately left alone, and how it was verified. These files are the record of *why* the project looks the way it does.
3. Before pushing, review the diff and then try to break it: run the change against real inputs, reproduce the bug before the fix, and check every claim in the docs against the code.
4. Open a pull request using the template and **squash-merge** it, so each PR becomes one commit on `main` with the PR title as its subject.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[(<scope>)]: <subject>

<body: what changed and why, wrapped at 72 characters>
```

- **type**: `feat` new behaviour · `fix` bug fix · `refactor` restructuring without behaviour change · `docs` · `test` · `perf` · `build` dependencies / build files · `chore` repo housekeeping
- **first line**: imperative, lower case, no trailing period, at most 72 characters including the type (`fix: correct class weight order`, not `Fixed class weights.`)
- **body**: explain why, not what the diff already shows; reference the plan doc for long reasoning
- Use English, one logical change per commit

## Pull requests

- **Title**: same format as a commit's first line, at most 65 characters: GitHub appends ` (#N)` when squashing, and the result becomes the commit on `main`. In the repo settings, set the squash commit message default to **Pull request title and description**.
- **Body**: follow `.github/pull_request_template.md` (Summary, Changes, Verification, Test plan).
