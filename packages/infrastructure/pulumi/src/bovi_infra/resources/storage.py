"""Storage resources: Storage Account and Azure Files share."""

from dataclasses import dataclass

import pulumi
import pulumi_azure_native.storage as storage

from ..types import ResourceTags


@dataclass
class StorageAccountArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    account_name: str
    tags: ResourceTags | None = None


@dataclass
class StorageAccountResult:
    account: storage.StorageAccount
    primary_key: pulumi.Output[str]
    connection_string: pulumi.Output[str]


def create_storage_account(name: str, args: StorageAccountArgs) -> StorageAccountResult:
    account = storage.StorageAccount(
        name,
        resource_group_name=args.resource_group_name,
        location=args.location,
        account_name=args.account_name,
        sku=storage.SkuArgs(name=storage.SkuName.STANDARD_LRS),
        kind=storage.Kind.STORAGE_V2,
        access_tier=storage.AccessTier.HOT,
        enable_https_traffic_only=True,
        minimum_tls_version=storage.MinimumTlsVersion.TLS1_2,
        allow_blob_public_access=False,
        tags=args.tags,
    )
    keys = storage.list_storage_account_keys_output(
        resource_group_name=args.resource_group_name,
        account_name=account.name,
    )
    primary_key = keys.keys[0].value
    connection_string = pulumi.Output.all(account.name, primary_key).apply(
        lambda a: (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={a[0]};"
            f"AccountKey={a[1]};"
            f"EndpointSuffix=core.windows.net"
        )
    )
    return StorageAccountResult(
        account=account,
        primary_key=primary_key,
        connection_string=connection_string,
    )


@dataclass
class FilesShareArgs:
    resource_group_name: pulumi.Input[str]
    account_name: pulumi.Input[str]
    share_name: str
    quota_gb: int = 1


@dataclass
class FilesShareResult:
    share: storage.FileShare


def create_files_share(name: str, args: FilesShareArgs) -> FilesShareResult:
    share = storage.FileShare(
        name,
        resource_group_name=args.resource_group_name,
        account_name=args.account_name,
        share_name=args.share_name,
        share_quota=args.quota_gb,
    )
    return FilesShareResult(share=share)


@dataclass
class BlobContainerArgs:
    resource_group_name: pulumi.Input[str]
    account_name: pulumi.Input[str]
    container_name: str


@dataclass
class BlobContainerResult:
    container: storage.BlobContainer


def create_blob_container(name: str, args: BlobContainerArgs) -> BlobContainerResult:
    container = storage.BlobContainer(
        name,
        resource_group_name=args.resource_group_name,
        account_name=args.account_name,
        container_name=args.container_name,
        public_access=storage.PublicAccess.NONE,
    )
    return BlobContainerResult(container=container)
