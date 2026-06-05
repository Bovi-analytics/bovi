"""Azure networking resource factories."""

from dataclasses import dataclass, field

import pulumi
import pulumi_azure_native.network as network

from ..types import ResourceTags


@dataclass
class SecurityRule:
    name: str
    priority: int
    protocol: str
    destination_port_range: str
    direction: str = "Inbound"
    access: str = "Allow"
    source_port_range: str = "*"
    source_address_prefix: str = "*"
    destination_address_prefix: str = "*"


@dataclass
class VirtualNetworkArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    vnet_name: str
    address_prefixes: list[str]
    subnet_name: str
    subnet_prefix: str
    tags: ResourceTags | None = None


@dataclass
class VirtualNetworkResult:
    vnet: network.VirtualNetwork
    subnet: network.Subnet


@dataclass
class PublicIpArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    name: str
    domain_name_label: str | None = None
    tags: ResourceTags | None = None


@dataclass
class NetworkInterfaceArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    name: str
    subnet_id: pulumi.Input[str]
    public_ip_id: pulumi.Input[str]
    network_security_group_id: pulumi.Input[str]
    tags: ResourceTags | None = None


@dataclass
class NetworkSecurityGroupArgs:
    resource_group_name: pulumi.Input[str]
    location: str
    name: str
    rules: list[SecurityRule] = field(default_factory=list)
    tags: ResourceTags | None = None


def create_virtual_network(name: str, args: VirtualNetworkArgs) -> VirtualNetworkResult:
    vnet = network.VirtualNetwork(
        name,
        resource_group_name=args.resource_group_name,
        virtual_network_name=args.vnet_name,
        location=args.location,
        address_space=network.AddressSpaceArgs(address_prefixes=args.address_prefixes),
        tags=args.tags,
    )
    subnet = network.Subnet(
        f"{name}-subnet",
        resource_group_name=args.resource_group_name,
        virtual_network_name=vnet.name,
        subnet_name=args.subnet_name,
        address_prefix=args.subnet_prefix,
    )
    return VirtualNetworkResult(vnet=vnet, subnet=subnet)


def create_public_ip(name: str, args: PublicIpArgs) -> network.PublicIPAddress:
    return network.PublicIPAddress(
        name,
        resource_group_name=args.resource_group_name,
        public_ip_address_name=args.name,
        location=args.location,
        public_ip_allocation_method="Static",
        sku=network.PublicIPAddressSkuArgs(name="Standard"),
        dns_settings=(
            network.PublicIPAddressDnsSettingsArgs(domain_name_label=args.domain_name_label)
            if args.domain_name_label
            else None
        ),
        tags=args.tags,
    )


def create_network_security_group(
    name: str,
    args: NetworkSecurityGroupArgs,
) -> network.NetworkSecurityGroup:
    return network.NetworkSecurityGroup(
        name,
        resource_group_name=args.resource_group_name,
        network_security_group_name=args.name,
        location=args.location,
        security_rules=[
            network.SecurityRuleArgs(
                name=rule.name,
                priority=rule.priority,
                direction=rule.direction,
                access=rule.access,
                protocol=rule.protocol,
                source_port_range=rule.source_port_range,
                destination_port_range=rule.destination_port_range,
                source_address_prefix=rule.source_address_prefix,
                destination_address_prefix=rule.destination_address_prefix,
            )
            for rule in args.rules
        ],
        tags=args.tags,
    )


def create_network_interface(name: str, args: NetworkInterfaceArgs) -> network.NetworkInterface:
    return network.NetworkInterface(
        name,
        resource_group_name=args.resource_group_name,
        network_interface_name=args.name,
        location=args.location,
        network_security_group=network.NetworkSecurityGroupArgs(id=args.network_security_group_id),
        ip_configurations=[
            network.NetworkInterfaceIPConfigurationArgs(
                name="internal",
                private_ip_allocation_method="Dynamic",
                subnet=network.SubnetArgs(id=args.subnet_id),
                public_ip_address=network.PublicIPAddressArgs(id=args.public_ip_id),
            )
        ],
        tags=args.tags,
    )
