"""Azure compute resource factories."""

import base64
from dataclasses import dataclass

import pulumi
import pulumi_azure_native.compute as compute

from ..types import ResourceTags


@dataclass
class LinuxVmArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    vm_name: str
    size: str
    admin_username: str
    ssh_public_key: pulumi.Input[str]
    network_interface_id: pulumi.Input[str]
    custom_data: pulumi.Input[str]
    tags: ResourceTags | None = None
    publisher: str = "Canonical"
    offer: str = "ubuntu-24_04-lts"
    sku: str = "server"
    version: str = "latest"
    os_disk_storage_account_type: str = "Standard_LRS"


def create_linux_vm(name: str, args: LinuxVmArgs) -> compute.VirtualMachine:
    encoded_custom_data = pulumi.Output.from_input(args.custom_data).apply(
        lambda value: base64.b64encode(value.encode()).decode()
    )
    return compute.VirtualMachine(
        name,
        resource_group_name=args.resource_group_name,
        vm_name=args.vm_name,
        location=args.location,
        hardware_profile=compute.HardwareProfileArgs(vm_size=args.size),
        network_profile=compute.NetworkProfileArgs(
            network_interfaces=[compute.NetworkInterfaceReferenceArgs(id=args.network_interface_id)]
        ),
        os_profile=compute.OSProfileArgs(
            computer_name=args.vm_name,
            admin_username=args.admin_username,
            custom_data=encoded_custom_data,
            linux_configuration=compute.LinuxConfigurationArgs(
                disable_password_authentication=True,
                ssh=compute.SshConfigurationArgs(
                    public_keys=[
                        compute.SshPublicKeyArgs(
                            path=f"/home/{args.admin_username}/.ssh/authorized_keys",
                            key_data=args.ssh_public_key,
                        )
                    ]
                ),
            ),
        ),
        storage_profile=compute.StorageProfileArgs(
            image_reference=compute.ImageReferenceArgs(
                publisher=args.publisher,
                offer=args.offer,
                sku=args.sku,
                version=args.version,
            ),
            os_disk=compute.OSDiskArgs(
                create_option="FromImage",
                caching=compute.CachingTypes.READ_WRITE,
                managed_disk=compute.ManagedDiskParametersArgs(
                    storage_account_type=args.os_disk_storage_account_type
                ),
            ),
        ),
        tags=args.tags,
    )
