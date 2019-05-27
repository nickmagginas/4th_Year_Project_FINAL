#include <string>

#include "ns3/core-module.h"
#include "ns3/global-route-manager.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor.h"
#include "ns3/log.h"

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("FirstSim");

int main(int argc, char *argv[]) {
    /*verbose_level = 0: Disable console logging
      verbose_level = 1: Simple console logging
      verbose_level = 2: Further console logging*/
    int verbose_level = 2;
    bool enableFlowMonitor = true;

    if (verbose_level == 1 || verbose_level == 2) {
        LogComponentEnable("FirstSim", LOG_LEVEL_INFO);
    } else if (verbose_level == 2) {
        //TODO: add more specific logging here
        LogComponentEnable("FirstSim", LOG_ALL);
    }
    
    //Set the next default to true if packet routing should be recomputed on an event such as
    //link being set to down:
    Config::SetDefault("ns3::Ipv4GlobalRouting::RespondToInterfaceEvents", BooleanValue(true));
    
    //Sets the default PacketSize and DataRate for the OnOffApplication interface:
    Config::SetDefault("ns3::OnOffApplication::PacketSize", UintegerValue(210));
    Config::SetDefault("ns3::OnOffApplication::DataRate", StringValue("300b/s"));

    CommandLine cmd;
    cmd.Parse (argc, argv);

    //Create 4 nodes in NodeContainer c:
    NS_LOG_INFO("Create nodes.");
    NodeContainer c;
    c.Create(4);
   
    std::string msg = "Number of nodes: " + std::to_string(c.GetN());
    NS_LOG_INFO(msg);

    //Set coordinate position of nodes in animation:
    AnimationInterface::SetConstantPosition(c.Get(0), 5, 5);
    AnimationInterface::SetConstantPosition(c.Get(1), 10, 10);
    AnimationInterface::SetConstantPosition(c.Get(2), 20, 9);
    AnimationInterface::SetConstantPosition(c.Get(3), 5, 30);

    //Create 4 new NodeContainers containing 2 nodes each,
    //these will define the links between nodes:
    NodeContainer n0n2 = NodeContainer (c.Get(0), c.Get(2));
    NodeContainer n1n2 = NodeContainer (c.Get(1), c.Get(2));
    NodeContainer n3n2 = NodeContainer (c.Get(3), c.Get(2));
    NodeContainer n1n3 = NodeContainer (c.Get(1), c.Get(3));
    
    //Define p2p links and their attributes between nodes: 
    NS_LOG_INFO("Create channels.");
    PointToPointHelper p2p;
    
    //Links d0d2 and d1d2 will have 5Mbps DataRate and 2ms Delay:
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    NetDeviceContainer d0d2 = p2p.Install(n0n2);
    NetDeviceContainer d1d2 = p2p.Install(n1n2);
    
    //Link d3d2 will have 1500kbps DataRate and 10ms Delay:
    p2p.SetDeviceAttribute("DataRate", StringValue("1500kbps"));
    p2p.SetChannelAttribute("Delay", StringValue("10ms"));
    NetDeviceContainer d3d2 = p2p.Install(n3n2);

    //Link d1d3 will have 1500kbps DataRate (From above ^) and 100ms Delay:
    p2p.SetChannelAttribute("Delay", StringValue("15ms"));
    NetDeviceContainer d1d3 = p2p.Install(n1n3);

    
    InternetStackHelper internet;
    //For each node in the input container c, 
    //implement ns3::Ipv4, ns3::Ipv6, ns3::Udp, and, ns3::Tcp classes: 
    internet.Install(c);

    //Assign IP addresses: 
    NS_LOG_INFO("Assign IP Addresses.");
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer i0i2 = ipv4.Assign(d0d2);

    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer i1i2 = ipv4.Assign(d1d2);

    ipv4.SetBase("10.2.2.0", "255.255.255.0");
    Ipv4InterfaceContainer i3i2 = ipv4.Assign(d3d2);

    ipv4.SetBase("10.3.3.0", "255.255.255.0");
    Ipv4InterfaceContainer i1i3 = ipv4.Assign(d1d3);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    //Create OnOff packet sender appilcation, to the ip address of node 3
    //sent from node 0.
    NS_LOG_INFO("Create Application.");
    uint16_t port = 9;

    OnOffHelper onoff ("ns3::UdpSocketFactory", 
            Address (InetSocketAddress (i1i3.GetAddress (1), port)));
    onoff.SetConstantRate (DataRate ("448kb/s"));
    ApplicationContainer apps = onoff.Install (c.Get (0));
    apps.Start (Seconds (1.0));
    apps.Stop (Seconds (10.0));

    // Create a packet sink to receive these packets on node 3
    PacketSinkHelper sink ("ns3::UdpSocketFactory",
            Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
    apps = sink.Install (c.Get (3));
    apps.Start (Seconds (1.0));
    apps.Stop (Seconds (10.0));

    // Create a similar flow from n3 to n1, starting at time 1.1 seconds
    onoff.SetAttribute ("Remote", AddressValue (InetSocketAddress (i1i2.GetAddress (0), port)));
    onoff.SetConstantRate(DataRate("300kb/s"));
    apps = onoff.Install (c.Get (3));
    apps.Start (Seconds (1.1));
    apps.Stop (Seconds (10.0));

    // Create a packet sink to receive these packets
    apps = sink.Install (c.Get (1));
    apps.Start (Seconds (1.1));
    apps.Stop (Seconds (10.0));

    //Schedule link between node 2 and 3 to be down between 3s and 5s:
    Ptr<Node> n3 = c.Get(3);
    Ptr<Ipv4> links_to_n3 = n3->GetObject<Ipv4>();
    //SetDown first link in links_to_n3 (which is link between 2 and 3):
    Simulator::Schedule(Seconds(3.0), &Ipv4::SetDown, links_to_n3, 1);
    //SetUp link between n2 and n3:
    Simulator::Schedule(Seconds(5.0), &Ipv4::SetUp, links_to_n3, 1);

    // Flow Monitor
    Ptr<FlowMonitor> flowMonitor;
    FlowMonitorHelper flowmonHelper; 
    if (enableFlowMonitor) {
        flowMonitor = flowmonHelper.InstallAll ();
    }

    //Enable ascii filestream of p2p trace:
    AsciiTraceHelper ascii;
    p2p.EnableAsciiAll (ascii.CreateFileStream ("firstsim/FirstSimTrace.tr"));
    p2p.EnablePcapAll ("firstsim/pcap/FirstSim");

    //Create xml animation file and routing table
    AnimationInterface anim("firstsim/FirstSimAnimation.xml");
    anim.EnablePacketMetadata(true);
    anim.EnableIpv4RouteTracking("firstsim/FirstRoutingTable.xml", 
            Seconds(0), Seconds(10), Seconds(0.25));

    NS_LOG_INFO ("Run Simulation.");

    Simulator::Stop(Seconds(11.0)); //required
    
    Simulator::Run();
    
    //if enableFlowMonitor = true, put flow monitor output in xml
    if (enableFlowMonitor) {
       flowMonitor->SerializeToXmlFile ("firstsim/FirstSimFlowMon.xml", true, true);
    }

    Simulator::Destroy();
    NS_LOG_INFO ("Done.");
    return 0;
}
