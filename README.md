# azure_ml_with_github_action

## Local setup

Create a `.env` file with:

```
AZUREML_SUBSCRIPTION_ID=...
AZUREML_RESOURCE_GROUP=...
AZUREML_WORKSPACE_NAME=...
AZUREML_DATA_URI=azureml://subscriptions/.../datastores/.../paths/.../data.csv
```

Then run:

```
python pipeline.py
```

## GitHub Actions

Configure these GitHub Secrets:

- `AZUREML_SUBSCRIPTION_ID`
- `AZUREML_RESOURCE_GROUP`
- `AZUREML_WORKSPACE_NAME`
- `AZUREML_DATA_URI`

On push to `main`, the workflow submits the pipeline using those secrets.