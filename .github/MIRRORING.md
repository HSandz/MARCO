# Push to both repositories

`HSandz/MARCO` is the source of truth. This checkout can push to both
`HSandz/MARCO` and `L2R-UET/MARCO` through `origin`, using the user's existing
GitHub SSH authentication. No deploy key, Actions secret, or background workflow
is required.

Configure each checkout once:

```sh
git config --local --replace-all remote.origin.pushurl git@github.com:HSandz/MARCO.git
git config --local --add remote.origin.pushurl git@github.com:L2R-UET/MARCO.git
```

Fetching from `origin` continues to use `HSandz/MARCO`. With the current branch
tracking `origin/main`, an ordinary `git push` sends that branch to both
repositories. Explicit branch pushes, tag pushes (`git push origin --tags`),
and deletions (`git push origin --delete BRANCH`) also go to both destinations.

This configuration is local and is not copied when someone clones the repo.
Changes made through GitHub or another checkout are not synchronized
automatically. Only the refs explicitly pushed are updated; extra remote refs
are not pruned. The two pushes are separate operations, so check both results
and retry if one destination fails.

Inspect the destinations with:

```sh
git remote get-url --push --all origin
```
