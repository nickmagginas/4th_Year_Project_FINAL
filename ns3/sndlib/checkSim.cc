#include "MySimulation.h"
int main(int argc, char const *argv[])
{
    mySimulation msim;
    msim.execute();
    //example printing out GlobalRouteLinkStateAdvertisement
    std::cout << msim.grlsaList[5] ;
    std::cout << msim.grlsaList[2] ;    

    uint16_t port = 9;

    OnOffHelper onoff ("ns3::UdpSocketFactory", 
            Address (InetSocketAddress (msim.ip[9].GetAddress (1), port)));
    onoff.SetConstantRate (DataRate ("448kb/s"));
    ApplicationContainer apps = onoff.Install (msim.container.Get (0));
    apps.Start (Seconds (1.0));
    apps.Stop (Seconds (10.0));

    // Create a packet sink to receive these packets on node 3
    PacketSinkHelper sink ("ns3::UdpSocketFactory",
            Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
    apps = sink.Install (msim.container.Get (3));
    apps.Start (Seconds (1.0));
    apps.Stop (Seconds (10.0));

    AnimationInterface anim("sndlibsim/FirstSimAnimation.xml");
    anim.EnablePacketMetadata(true);
    anim.EnableIpv4RouteTracking("sndlibsim/FirstRoutingTable.xml", 
            Seconds(0), Seconds(10), Seconds(0.25));

    Simulator::Stop(Seconds(11.0)); //required
    
    Simulator::Run();

    std::cout<<"run";

    Simulator::Destroy();


    return 0;
}
