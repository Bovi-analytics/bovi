"""Composed public Linux VM component."""

from dataclasses import dataclass

import pulumi
import pulumi_azure_native.network as network

from bovi_infra.resources.compute import LinuxVmArgs, create_linux_vm
from bovi_infra.resources.network import (
    NetworkInterfaceArgs,
    NetworkSecurityGroupArgs,
    PublicIpArgs,
    SecurityRule,
    create_network_interface,
    create_network_security_group,
    create_public_ip,
)
from bovi_infra.types import ResourceTags


@dataclass
class PublicLinuxVmArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    subnet_id: pulumi.Input[str]
    vm_name: str
    public_ip_name: str
    network_security_group_name: str
    network_interface_name: str
    ssh_public_key: pulumi.Input[str]
    custom_data: pulumi.Input[str]
    security_rules: list[SecurityRule]
    admin_username: str = "adminuser"
    vm_size: str = "Standard_B2ats_v2"
    domain_name_label: str | None = None
    tags: ResourceTags | None = None


@dataclass
class PublicLinuxVmResult:
    public_ip: network.PublicIPAddress
    network_security_group: network.NetworkSecurityGroup
    network_interface: network.NetworkInterface
    virtual_machine: object
    fqdn: pulumi.Output[str]


def create_public_linux_vm(name: str, args: PublicLinuxVmArgs) -> PublicLinuxVmResult:
    public_ip = create_public_ip(
        f"{name}-pip",
        PublicIpArgs(
            resource_group_name=args.resource_group_name,
            location=args.location,
            name=args.public_ip_name,
            domain_name_label=args.domain_name_label,
            tags=args.tags,
        ),
    )
    nsg = create_network_security_group(
        f"{name}-nsg",
        NetworkSecurityGroupArgs(
            resource_group_name=args.resource_group_name,
            location=args.location,
            name=args.network_security_group_name,
            rules=args.security_rules,
            tags=args.tags,
        ),
    )
    nic = create_network_interface(
        f"{name}-nic",
        NetworkInterfaceArgs(
            resource_group_name=args.resource_group_name,
            location=args.location,
            name=args.network_interface_name,
            subnet_id=args.subnet_id,
            public_ip_id=public_ip.id,
            network_security_group_id=nsg.id,
            tags=args.tags,
        ),
    )
    vm = create_linux_vm(
        name,
        LinuxVmArgs(
            resource_group_name=args.resource_group_name,
            location=args.location,
            vm_name=args.vm_name,
            size=args.vm_size,
            admin_username=args.admin_username,
            ssh_public_key=args.ssh_public_key,
            network_interface_id=nic.id,
            custom_data=args.custom_data,
            tags=args.tags,
        ),
    )
    fqdn = public_ip.dns_settings.apply(
        lambda settings: settings.fqdn if settings and settings.fqdn else ""
    )
    return PublicLinuxVmResult(
        public_ip=public_ip,
        network_security_group=nsg,
        network_interface=nic,
        virtual_machine=vm,
        fqdn=fqdn,
    )
