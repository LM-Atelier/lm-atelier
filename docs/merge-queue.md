# Merging changes into develop

Open a focused pull request against `develop` and complete its required checks.
Once the pull request is ready to merge, add it to GitHub's merge queue. The queue
creates an integration commit containing the current `develop` branch, earlier
queued changes, and the pull request. Required checks run on that combined commit
before GitHub lands the change.

A pull request does not need a rebase merely because `develop` advanced. The queue
checks the updated combination. Resolve merge conflicts on the work branch; new
commits need fresh pull-request checks before they can enter the queue.

If checks fail or GitHub removes a pull request from the queue, inspect the failed
run or removal reason. Correct the problem, wait for any required checks, and add
the pull request to the queue again. A successful run on an earlier combination
is not a substitute for the current queue checks.

The queue uses squash merges (`SQUASH`) and requires every integration commit to
pass (`ALLGREEN`). It allows three concurrent verification builds, lands one pull
request at a time, and gives required checks 60 minutes to report a conclusion.
Each landed commit uses the pull-request title and number as its subject, with an
empty commit body. Keep the title descriptive and put the explanation and
validation details in the pull-request description.

Promotions from protected `develop` to `main` keep their existing exact-tree
promotion checks. The queue applies only to `develop`.

Repository administrators can apply the queue configuration after its CI workflow
and runtime scripts have been deployed on `develop`:

```powershell
.\scripts\configure-public-repository.ps1 -Apply -MergeQueueOnly
```

The command checks that the deployed verification files match the local files,
then applies and verifies the develop queue and squash-commit settings.
