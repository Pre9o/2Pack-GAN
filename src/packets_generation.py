from genericpath import exists
from decoder_pac_gan_dns_ip import decode_packets
from gan_packets_generator import generate_packet_by_gan
import sys
import binascii
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
generated_packets_dir = os.path.join(current_dir, 'generated_dns_ip_packets_by_gan')

DNS_IP_dir = os.path.dirname(current_dir)
PAC_GAN_dir = os.path.dirname(DNS_IP_dir,)
FINE_TUNNING_dir = os.path.join(PAC_GAN_dir, 'FINE-TUNING')
NOVO_FINE_TUNING_dir = os.path.join(FINE_TUNNING_dir, 'NOVO FINE-TUNING')
GAN_dir = os.path.join(DNS_IP_dir, 'GAN')
output_images_dir = os.path.join(GAN_dir, 'output_images')

if not os.path.exists(generated_packets_dir):
    os.makedirs(generated_packets_dir)

pcap_global_header = ('D4 C3 B2 A1'   
                      '02 00'         #File format major revision (i.e. pcap <2>.4)  
                      '04 00'         #File format minor revision (i.e. pcap 2.<4>)   
                      '00 00 00 00'     
                      '00 00 00 00'     
                      'FF FF 00 00'     
                      '01 00 00 00')

pcap_packet_header = ('AA 77 9F 47'     
                      '90 A2 04 00'     
                      'XX XX XX XX'   #Frame Size (little endian) 
                      'YY YY YY YY')  #Frame Size (little endian)

eth_header = ('00 00 00 00 00 00'     #Source Mac    
              '00 00 00 00 00 00'     #Dest Mac  
              '08 00')                #Protocol (0x0800 = IP)
                
def getByteLength(str1):
    '''Returns the length of a string in bytes
    
    Args:
        str1 (string): String
        
    Returns:
        integer: Length of the string in bytes
    '''
    return int(len(''.join(str1.split())) / 2)

def writeByteStringToFile(bytestring, filename):
    '''Writes a bytestring to a file
    
    Args:
        bytestring (string): Bytestring to be written to the file
        filename (string): Name of the file to be written
    '''
    bytelist = bytestring.split()  
    bytes = binascii.a2b_hex(''.join(bytelist))
    bitout = open(filename, 'wb')
    bitout.write(bytes)
    
def splitN(str1, n):
    '''Splits the string into a list of tokens every n characters
    
    Args:
        str1 (string): String to be split
        n (integer): Number of characters to split the string
        
    Returns:
        list: List of tokens
    '''
    return [str1[start:start+n] for start in range(0, len(str1), n)]
    
def ip_checksum(ip):
    '''Calculates and returns the IP checksum based on the given IP Header
    
    Args:
        ip (string): IP Header
        
    Returns:
        integer: IP checksum
            
    '''
    words = splitN(''.join(ip.split()),4)

    csum = 0
    for word in words:
        csum += int(word, base=16)

    csum += (csum >> 16)
    csum = csum & 0xFFFF ^ 0xFFFF

    return csum

def protocol_checksum(protocol):
    '''Calculates and returns the protocol checksum based on the given protocol Header
    
    Args:
        protocol (string): protocol Header
        
    Returns:
        integer: protocol checksum
            
    '''
    words = splitN(''.join(protocol.split()),4)

    csum = 0
    for word in words:
        csum += int(word, base=16)

    csum += (csum >> 16)
    csum = csum & 0xFFFF ^ 0xFFFF

    return csum

def generatePcapFile(output_path, number_of_packets, ipv4_header, protocol_header_data, training=False, epoch_path=None):
    '''Generates a pcap file based on the given filename
    
    Args:
        filename (string): Name of the pcap file to be generated
        
    Returns:
        string: Bytestring of the generated pcap file
    '''
    for i in range(1, number_of_packets + 1):
        protocol_len = getByteLength(protocol_header_data[i-1])
        protocol = protocol_header_data[i-1].replace('XXXX',"%04x"%protocol_len)
        checksum = protocol_checksum(protocol.replace('YYYY','00 00'))
        protocol = protocol.replace('YYYY',"%04x"%checksum)
        
        ip_len = protocol_len + getByteLength(ipv4_header[i-1])
        ip = ipv4_header[i-1].replace('XXXX',"%04x"%ip_len)
        checksum = ip_checksum(ip.replace('YYYY','00 00'))
        ip = ip.replace('YYYY',"%04x"%checksum)
        
        pcap_len = ip_len + getByteLength(eth_header)
        hex_str = "%08x"%pcap_len
        reverse_hex_str = hex_str[6:] + hex_str[4:6] + hex_str[2:4] + hex_str[:2]
        pcaph = pcap_packet_header.replace('XX XX XX XX',reverse_hex_str)
        pcaph = pcaph.replace('YY YY YY YY',reverse_hex_str)
            
        if i == 1:
            bytestring = pcap_global_header + pcaph + eth_header + ip + protocol
        else:
            bytestring += pcaph + eth_header + ip + protocol
    
    writeByteStringToFile(bytestring, output_path)
    

def save_packets_on_training(images, output_path, epoch, examples):
    '''Saves packets during training
    
    Args:
        images: Generated images from GAN
        filename: Base filename for the pcap file
        epoch: Current training epoch
        examples: Number of examples
    '''
    epoch_path = os.path.join(output_images_dir, f"generated_images_{epoch}")
    
    ipv4_header = []
    protocol_header_data = []
    
    for i in range(1, examples + 1):
        raw_ipv4, raw_protocol = decode_packets(images[i-1])

        ipv4_header.append(raw_ipv4[0:4] + 'XXXX' + raw_ipv4[8:20] + 'YYYY' + raw_ipv4[24:40])
                    
        if len(raw_ipv4) > 18 and raw_ipv4[18:20] == '11':
                # print(f"Raw protocol for UDP: {raw_protocol}")
                protocol_header_data.append(raw_protocol[0:8] + 'XXXX' + 'YYYY' + raw_protocol[16:])
        else:
            protocol_header_data.append(raw_protocol[0:4] + 'XXXX' + raw_protocol[8:16] + raw_protocol[48:128] if len(raw_protocol) > 128 else raw_protocol[8:]) # ICMP
            
    generatePcapFile(output_path, examples, ipv4_header, protocol_header_data, True, epoch_path)
    print(f"Pcap file generated! {output_path}")
    
def main():
    '''Main function to generate packets and pcap file'''
    number_of_packets = int(input("Type the number of packets to generate: "))
    
    generator_option = input("Choose the generator model: \n1 - Generator model \n2 - Generator model FT\n")

    ipv4_header = []
    protocol_header_data = []

    raw_bytes = generate_packet_by_gan(generator_option, number_of_packets)
    
    for i in range(1, number_of_packets + 1):
        raw_ipv4, raw_protocol = decode_packets(raw_bytes[i-1])

        ipv4_header.append(raw_ipv4[0:4] + 'XX' 'XX' + raw_ipv4[8:20] + 'YY' 'YY' + raw_ipv4[24:40])
                
        if ipv4_header[i-1][9:11] == '11':
            protocol_header_data.append(raw_protocol[0:4] + 'XX' 'XX' + raw_protocol[8:16] + raw_protocol[48:128])
        else:
            protocol_header_data.append(raw_protocol[0:8] + 'XX' 'XX' + 'YY' 'YY' + raw_protocol[16:])
            
        print("Packet " + str(i) + " generated!")

   
    pcap_name = input("Type the name of the pcap file: ")        
    pcapfile = pcap_name + '.pcap'

    generatePcapFile(pcapfile, number_of_packets, ipv4_header, protocol_header_data)
    print("Pcap file generated!")
    
if __name__ == '__main__':
    main()