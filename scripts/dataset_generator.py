import os
import pandas as pd
import json
from nslookup import Nslookup

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_directory)
pcap_dir = os.path.join(parent_dir, 'PCAPS')

csv_file_path = os.path.join(current_directory, 'top500Domains.csv')

ud_ip_file_path = os.path.join(current_directory, 'unreachable_domains_ip.json')
ud_dns_file_path = os.path.join(current_directory, 'unreachable_domains_dns.json')

if not os.path.exists(ud_ip_file_path):
    with open(ud_ip_file_path, 'w') as f:
        json.dump([], f)
        
if not os.path.exists(ud_dns_file_path):
    with open(ud_dns_file_path, 'w') as f:
        json.dump([], f)

current_unreachable_domains = []

df = pd.DataFrame(pd.read_csv(csv_file_path, sep=','))
print(df.head()) 

def detect_unreachable_domains(option, num_requests=1, dns_query = Nslookup(dns_servers=["1.1.1.1"], verbose=False, tcp=False)):
    """Detects unreachable domains using ICMP or DNS requests

    Args:
        option (integer): 1 for ICMP, 2 for DNS
        num_requests (int, optional): It is 1, because this is just a verification. Defaults to 1.
        dns_query (_type_, optional): Query for nslookup. Defaults to Nslookup(dns_servers=["1.1.1.1"], verbose=False, tcp=False).
    """
    if option == '1':
        for index, row in df.iterrows():
            if ping_ip(row['Root Domain'], num_requests):  # ICMP
                print(f'ICMP: {index} working')
            else:
                print(f'ICMP: {index} not working')
                current_unreachable_domains.append(row['Root Domain'])
        
        with open(ud_ip_file_path, 'w') as f:
            json.dump(current_unreachable_domains, f)
                
    elif option == '2':
        for index, row in df.iterrows():
            if nslookup_request(row['Root Domain'], num_requests, dns_query):  # DNS
                print(f'DNS: {index} working')
            else:
                print(f'DNS: {index} not working')
                current_unreachable_domains.append(row['Root Domain'])
                
        with open(ud_dns_file_path, 'w') as f:
            json.dump(current_unreachable_domains, f)         


def icmp_capture(pcap_name, num_pings, pcap_folder):
    """Captures ICMP packets

    Args:
        pcap_name (string): Name of the pcap file
        num_pings (integer): Number of pings
    """
    os.system(f"sudo tcpdump -w {os.path.join(pcap_folder, pcap_name)} -i eth0 icmp &")  # Start the ICMP capture process
    unreachable_domains_file_ip = json.load(open(ud_ip_file_path))
    
    for index, row in df.iterrows():
        if row['Root Domain'] in unreachable_domains_file_ip:
            continue
        else:
            if ping_ip(row['Root Domain'], num_pings):  # ICMP
                print(row['Root Domain'])
            else:
                print('Could not ping/DNS the domain: ' + row['Root Domain'])
        
    os.system("sudo pkill -2 tcpdump")  # Kill the process
    

def dns_capture(pcap_name, num_dns, pcap_folder):
    """Captures DNS packets

    Args:
        pcap_name (string): Name of the pcap file
        num_dns (integer): Number of DNS requests
    """
    os.system(f"sudo tcpdump -w {os.path.join(pcap_folder, pcap_name)} -i eth0 udp dst port 53 &")  # Start the DNS capture process
    unreachable_domains_file_dns = json.load(open(ud_dns_file_path))
    
    dns_query = Nslookup(dns_servers=["1.1.1.1"], verbose=False, tcp=False)
    
    for index, row in df.iterrows():
        if row['Root Domain'] in unreachable_domains_file_dns:
            continue
        else:
            if nslookup_request(row['Root Domain'], num_dns, dns_query):  # DNS
                print(row['Root Domain'])
            else:
                print('Could not ping/DNS the domain: ' + row['Root Domain'])
        
    os.system("sudo pkill -2 tcpdump")  # Kill the process
    
    
def icmp_dns_capture(pcap_name, num_requests, pcap_folder):
    """Captures ICMP and DNS packets

    Args:
        pcap_name (string): Name of the pcap file
        num_requests (integer): Number of requests
    """
    os.system(f"sudo tcpdump -w {os.path.join(pcap_folder, pcap_name)} -i eth0 icmp or udp dst port 53 &")  # Start the ICMP and DNS capture process
    unreachable_domains_file_ip = json.load(open(ud_ip_file_path))
    unreachable_domains_file_dns = json.load(open(ud_dns_file_path))
    
    dns_query = Nslookup(dns_servers=["1.1.1.1"], verbose=False, tcp=False)
    
    for index, row in df.iterrows():
        if row['Root Domain'] in unreachable_domains_file_ip or row['Root Domain'] in unreachable_domains_file_dns:
            continue
        else:
            if ping_ip(row['Root Domain'], num_requests) and nslookup_request(row['Root Domain'], num_requests, dns_query):  # ICMP and DNS
                print(row['Root Domain'])
            else:
                print('Could not ping/DNS the domain: ' + row['Root Domain'])
                
    os.system("sudo pkill -2 tcpdump")  # Kill the process
    

def ping_ip(domain, num_pings):  # ICMP
    """Pings a domain

    Args:
        domain (string): Domain to be pinged
        num_pings (integer): Number of pings

    Returns:
        boolean: True if the domain is reachable, False otherwise
    """
    response = os.system(f"ping -c {num_pings} {domain}")
    if response == 0:
        return True
    else:
        return False
    

def nslookup_request(domain, num_dns, dns_query):  # DNS
    """Makes a DNS request

    Args:
        domain (string): Domain to be requested
        num_dns (integer): Number of DNS requests
        dns_query (nslookup): Query for nslookup

    Returns:
        boolean: True if the domain is reachable, False otherwise
    """
    for i in range(0, num_dns):
        ips_record = dns_query.dns_lookup(domain)
        print(ips_record.response_full, ips_record.answer)

    if ips_record.answer is not None:
        return True
    else:
        return False


def main():
    """Main function
    """
    


    while True:
        option = input("Choose the option:\n1 - ICMP\n2 - DNS\n3 - ICMP and DNS\n")
        
        if option == '1':
            num_pings = int(input("Enter the number of pings: "))
            pcap_name = input("Enter the name of the pcap file: ") + '.pcap'
            detect_unreachable_domains(option)
            icmp_capture(pcap_name, num_pings, pcap_dir)
            break
        
        elif option == '2':
            num_dns = int(input("Enter the number of dns requests: "))
            pcap_name = input("Enter the name of the pcap file: ") + '.pcap'
            detect_unreachable_domains(option)
            dns_capture(pcap_name, num_dns, pcap_dir)
            break
        
        elif option == '3':
            num_requests = int(input("Enter the number of requests: "))
            pcap_name = input("Enter the name of the pcap file: ") + '.pcap'
            detect_unreachable_domains(option, num_requests)
            icmp_dns_capture(pcap_name, num_requests, pcap_dir)
            break
        
        else:
            print('Invalid option!')
            
    
if __name__ == '__main__':
    """Starts the program
    """
    main()
