# How to refactor?

### Important to Note

1. Run the `application.py` after performing any type modification because it works like application testing.

```sh
$ http://localhost:5000/
```

2. Do one at once because if you don't it mess things up to keep records of modifications.

## Step 1 - Basic

- [x] First run the app.
- [x] Remove unnecessary files.
- [x] Format the scripts.

### Step 1.1 - Exception and Logging

- [x] Refactor **exceptions**.
- [x] Refactor **logging**.
  > I don't know what to do about it. **I didn't do anything.**

## Step 2 - 1-on-1

> Refactor file to file from **ingestion to model building**.

- [x] `application`
- [x] `backorder.constant`
- [x] `backorder.entity`
- [ ] `backorder.config`
- [ ] `backorder.component.data_ingestion`
- [ ] `backorder.component.data_transformation`
- [ ] `backorder.component.data_validation`
- [ ] `backorder.component.model_trainer`
- [ ] `backorder.component.model_pusher`
- [ ] `backorder.component.model_evaluation`
