"""
Dataset generator for capturing real network traffic.

This script captures ICMP and/or DNS traffic from a list of domains
to create datasets for training the GAN. It uses tcpdump to capture
packets and can generate ICMP, DNS, or combined traffic datasets.
"""

import os
import pandas as pd
from nslookup import Nslookup
from argparse import ArgumentParser


DEFAULT_DNS_SERVER = "1.1.1.1"
DEFAULT_INTERFACE = "eth0"

class DatasetConfig:
    """Manages directory paths for dataset generation."""
    
    def __init__(self, base_dir=None):
        """Initialize dataset configuration.
        
        Args:
            base_dir (str, optional): Base directory. Defaults to parent of script directory
        """
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
        
        self.base_dir = base_dir
        self.pcaps_dir = os.path.join(base_dir, 'pcaps')
        self.data_dir = os.path.join(base_dir, 'data')
        self.csv_dir = os.path.join(self.data_dir, 'csv')
        
        os.makedirs(self.pcaps_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
    
    def get_csv_path(self, csv_name):
        """Get full path to CSV file.
        
        Args:
            csv_name (str): Name of CSV file
            
        Returns:
            str: Full path to CSV file
        """
        return os.path.join(self.csv_dir, csv_name)
    
    def get_pcap_path(self, pcap_name):
        """Get full path to PCAP file.
        
        Args:
            pcap_name (str): Name of PCAP file
            
        Returns:
            str: Full path to PCAP file
        """
        if not pcap_name.endswith('.pcap'):
            pcap_name += '.pcap'
        return os.path.join(self.pcaps_dir, pcap_name)


def ping_domain(domain, num_pings):
    """Ping a domain using ICMP.

    Args:
        domain (str): Domain to ping
        num_pings (int): Number of ping requests

    Returns:
        bool: True if domain is reachable, False otherwise
    """
    response = os.system(f"ping -c {num_pings} {domain} > /dev/null 2>&1")
    return response == 0


def nslookup_domain(domain, num_requests, dns_query):
    """Perform DNS lookups for a domain.

    Args:
        domain (str): Domain to lookup
        num_requests (int): Number of DNS requests
        dns_query (Nslookup): Nslookup instance

    Returns:
        bool: True if domain is resolvable, False otherwise
    """
    answer = None
    
    for _ in range(num_requests):
        try:
            result = dns_query.dns_lookup(domain)
            answer = result.answer
            if answer:
                print(f"Resolved: {domain} → {answer}")
        except Exception as e:
            print(f"DNS error for {domain}: {e}")
    
    return answer is not None


def icmp_capture(domains_df, pcap_path, num_pings, interface=DEFAULT_INTERFACE):
    """Capture ICMP (ping) packets from domains list.

    Args:
        domains_df (pd.DataFrame): DataFrame with 'Root Domain' column
        pcap_path (str): Full path to output PCAP file
        num_pings (int): Number of pings per domain
        interface (str): Network interface to capture on
    """
    print(f"Starting tcpdump on interface {interface}...")
    
    os.system(f"sudo tcpdump -w {pcap_path} -i {interface} icmp > /dev/null 2>&1 &")
    
    successful = 0
    failed = 0
    
    for index, row in domains_df.iterrows():
        domain = row['Root Domain']
        print(f"Pinging {domain}...\n")
        
        if ping_domain(domain, num_pings):
            print(f"{domain} is reachable")
            successful += 1
        else:
            print(f"{domain} is unreachable")
            failed += 1
    
    print(f"Stopping capture...\n")
    os.system("sudo pkill -2 tcpdump")
    
    print(f"ICMP capture complete: {successful} successful, {failed} failed\n")
    

def dns_capture(domains_df, pcap_path, num_requests, interface=DEFAULT_INTERFACE, dns_server=DEFAULT_DNS_SERVER):
    """Capture DNS packets from domains list.

    Args:
        domains_df (pd.DataFrame): DataFrame with 'Root Domain' column
        pcap_path (str): Full path to output PCAP file
        num_requests (int): Number of DNS requests per domain
        interface (str): Network interface to capture on
        dns_server (str): DNS server to use for lookups
    """
    print(f"Starting tcpdump on interface {interface}...")
    
    os.system(f"sudo tcpdump -w {pcap_path} -i {interface} 'udp dst port 53' > /dev/null 2>&1 &")
    
    dns_query = Nslookup(dns_servers=[dns_server], verbose=False, tcp=False)
    
    successful = 0
    failed = 0
    
    for index, row in domains_df.iterrows():
        domain = row['Root Domain']
        print(f"Querying {domain}...\n")
        
        if nslookup_domain(domain, num_requests, dns_query):
            successful += 1
        else:
            print(f"{domain} could not be resolved")
            failed += 1
    
    print(f"Stopping capture...\n")
    os.system("sudo pkill -2 tcpdump")
    
    print(f"DNS capture complete: {successful} successful, {failed} failed\n")
    
def icmp_dns_capture(domains_df, pcap_path, num_requests, interface=DEFAULT_INTERFACE, dns_server=DEFAULT_DNS_SERVER):
    """Capture both ICMP and DNS packets from domains list.

    Args:
        domains_df (pd.DataFrame): DataFrame with 'Root Domain' column
        pcap_path (str): Full path to output PCAP file
        num_requests (int): Number of requests per domain (both ICMP and DNS)
        interface (str): Network interface to capture on
        dns_server (str): DNS server to use for lookups
    """
    print(f"\n[ICMP + DNS CAPTURE]")
    print(f"Starting tcpdump on interface {interface}...")
    
    os.system(f"sudo tcpdump -w {pcap_path} -i {interface} 'icmp or (udp dst port 53)' > /dev/null 2>&1 &")
    
    dns_query = Nslookup(dns_servers=[dns_server], verbose=False, tcp=False)
    
    successful = 0
    failed = 0
    
    for index, row in domains_df.iterrows():
        domain = row['Root Domain']
        print(f"Testing {domain}...\n")
        
        ping_ok = ping_domain(domain, num_requests)
        dns_ok = nslookup_domain(domain, num_requests, dns_query)
        
        if ping_ok and dns_ok:
            print(f"{domain} is fully reachable (ICMP + DNS)")
            successful += 1
        else:
            print(f"{domain} partially reachable (ICMP: {ping_ok}, DNS: {dns_ok})")
            failed += 1
    
    print(f"Stopping capture...\n")
    os.system("sudo pkill -2 tcpdump")
    
    print(f"ICMP+DNS capture complete: {successful} fully reachable, {failed} partial/failed\n")


def main():
    """Main function to generate network traffic dataset.
    
    This script requires sudo privileges to run tcpdump.
    Ensure the CSV file contains a 'Root Domain' column with domain names.
    """
    args_parser = ArgumentParser(description='Generate network traffic dataset by capturing ICMP/DNS packets')
    args_parser.add_argument('--csv_name',type=str,required=True,help='Name of the CSV file with domains (must have "Root Domain" column)')
    args_parser.add_argument('--protocol',type=str,choices=['icmp', 'dns', 'icmp_dns'],default='icmp',help='Protocol to capture: icmp (ping), dns (lookups), or icmp_dns (both)')
    args_parser.add_argument('--num_requests',type=int,default=5,help='Number of requests per domain (default: 5)')
    args_parser.add_argument('--pcap_name',type=str,required=True,help='Name of the output PCAP file')
    args_parser.add_argument('--interface',type=str,default=DEFAULT_INTERFACE,help=f'Network interface to capture on (default: {DEFAULT_INTERFACE})')
    args_parser.add_argument('--dns_server',type=str,default=DEFAULT_DNS_SERVER,help=f'DNS server for lookups (default: {DEFAULT_DNS_SERVER})')
    args = args_parser.parse_args()
    
    config = DatasetConfig()
    
    print(f"Protocol: {args.protocol.upper()}")
    print(f"Requests per domain: {args.num_requests}")
    print(f"Interface: {args.interface}")
    
    csv_path = config.get_csv_path(args.csv_name)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}\n")
        return
    
    try:
        domains_df = pd.read_csv(csv_path)
        
        if 'Root Domain' not in domains_df.columns:
            print(f"Error: CSV must contain 'Root Domain' column\n")
            print(f"Available columns: {', '.join(domains_df.columns)}")
            return
        
        print(f"Loaded {len(domains_df)} domains from {args.csv_name}\n")
        
    except Exception as e:
        print(f"Error loading CSV: {e}\n")
        return
    
    pcap_path = config.get_pcap_path(args.pcap_name)
    print(f"Output PCAP: {pcap_path}\n")
    
    try:
        if args.protocol == 'icmp':
            icmp_capture(domains_df, pcap_path, args.num_requests, args.interface)
        
        elif args.protocol == 'dns':
            dns_capture(domains_df, pcap_path, args.num_requests, args.interface, args.dns_server)
        
        elif args.protocol == 'icmp_dns':
            icmp_dns_capture(domains_df, pcap_path, args.num_requests, args.interface, args.dns_server)
        
        print(f"PCAP file saved to: {pcap_path}")
    
        
    except KeyboardInterrupt:
        print(f"Capture interrupted by user\n")
        print(f"Stopping tcpdump...")
        os.system("sudo pkill -2 tcpdump")
        print(f"Partial capture saved to: {pcap_path}\n")
    
    except Exception as e:
        print(f"Error during capture: {e}\n")
        os.system("sudo pkill -2 tcpdump")
    
if __name__ == '__main__':
    main()
