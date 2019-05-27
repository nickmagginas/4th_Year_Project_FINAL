//#define _GLIBCXX_USE_CXX11_ABI 0
#include <string>
#include <tuple>
#include <map>
#include <iostream>
#include "parser.h"

#include "ns3/vector.h"
#include "ns3/ptr.h"
#include "ns3/core-module.h"
#include "ns3/global-route-manager.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-grid.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/flow-monitor.h"
#include "ns3/log.h"
#include "ns3/geographic-positions.h"
#include "ns3/position-allocator.h"
#include "ns3/names.h"

#include "ns3/assert.h"
#include "ns3/simulation-singleton.h"
#include "ns3/global-route-manager-impl.h"


using namespace ns3;
NS_LOG_COMPONENT_DEFINE("FirstSim");


int main(int argc, char *argv[]) {
     //create an instance of Parser
     Parser g50;
     //read germany50 file
     g50.read_file();

     //create a map nodes with keys of type string and values of type tuple(string,string)
     std::map<std::string, std::tuple<std::string, std::string>> nodes;
     //from parser.h this function will return nodes into map nodes exp : Aachen ( 6.04 50.76 )
     nodes = g50.get_nodes();
     //create an iterator for the map nodes
     std::map<std::string, std::tuple<std::string, std::string>>::iterator nodeIt;

     //create a map links with keys of type string and values of type tuple(string,string)   
     std::map<std::string, std::tuple<std::string, std::string, std::string, std::string>> links;
     //from parser.h, return link into map links 
     links = g50.get_links();
     //create an iterator for the map links
     std::map<std::string, std::tuple<std::string, std::string, std::string, std::string>>::iterator linkIt;

     /*start the iteration for every element in map nodes and
       print key and value in the map nodes
     for (nodeIt = nodes.begin(); nodeIt != nodes.end(); ++nodeIt)
     {
	    std::cout << nodeIt->first << " => " << std::get<0>(nodeIt->second) << " " << std::get<1>(nodeIt->second) << '\n';
     }
     */
    
/////////////////////////////////////////////////////////////////////////////////////
    /*verbose_level = 0: Disable console logging
      verbose_level = 1: Simple console logging
      verbose_level = 2: Further console logging*/
    int verbose_level = 2;
    //bool enableFlowMonitor = true;

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
    
    //Create nodes in NodeContainer c:
    NS_LOG_INFO("Create nodes.");
    NodeContainer c;
    //get the size of nodes
    int numNodes = nodes.size();
    c.Create(numNodes);
   
    std::string msg = "Number of nodes: " + std::to_string(c.GetN());
    NS_LOG_INFO(msg);

/////////////////////////////////////////////////////////////////////////////////////

    //Set coordinate position of nodes in animation:
    int i = 0;
    for (auto nodeIt = nodes.begin(); nodeIt != nodes.end(); ++nodeIt) {
        Ptr<Node> node = c.Get(i);

	    double latitude = std::stod(std::get<0>(nodeIt->second));
        double longitude = std::stod(std::get<1>(nodeIt->second));
        //std::cout<<latitude<<" "<<longitude<<std::endl;
    	
	/*Converts earth geographic/geodetic coordinates (latitude and longitude in degrees) with a given altitude above earth's surface 
	  (in meters) to Earth Centered Earth Fixed (ECEF) Cartesian coordinates (x, y, z in meters), where origin (0, 0, 0) is the center of 		  the earth.*/
	//using GeographicToCartesianCoordinates()

        Names::Add(nodeIt->first, node);
        Vector coordinates = GeographicPositions::GeographicToCartesianCoordinates(
                latitude, longitude, 0, GeographicPositions::SPHERE);
        //std::cout<<" Node: "<<i<<" "<coordinates<<std::endl;
        float x = (coordinates.x); 
        float y = (coordinates.y);

        //std::cout<< nodeIt->first <<" Node:"<<i<<" x: "<<x<<" | y: "<<y<<std::endl;
	
	//plot the node 
        AnimationInterface::SetConstantPosition(node, x, y);        
        ++i;
    }


///////////////////////////////////////////////////////////////////////////////

	//connecting the nodes from source to target

    std::cout<<"GOT TO NODECONTAINERLIST LOOP"<<std::endl;
    //create an empty nodeContainerList
    std::vector<NodeContainer> nodeContainerList(links.size());

    i = 0;
    for (auto linkIt = links.begin(); linkIt != links.end(); ++linkIt) {
        std::string source = std::get<0>(linkIt->second);
        std::string target = std::get<1>(linkIt->second);
       
        //std::cout<<"Loop :"<<i<<" | source: "<<source<<" | target: "<<target<<std::endl;

        Ptr<Node> sourceNode (Names::Find<Node>(source));
        Ptr<Node> targetNode (Names::Find<Node>(target));
        
        nodeContainerList[i] = NodeContainer(sourceNode, targetNode);
        
        ++i;
    }

    std::cout<<i<<" nodeContainerList size: "<<nodeContainerList.size()<<std::endl;

//////////////////////////////////////////////////////////////////////////////

    //Define p2p links and their attributes between nodes: 
    NS_LOG_INFO("Create channels.");
    PointToPointHelper p2p;
    
    //Links d0d2 and d1d2 will have 5Mbps DataRate and 2ms Delay:
    p2p.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));
    
    std::vector<NetDeviceContainer> netDeviceContainerList(nodeContainerList.size());
    for(unsigned int i=0; i<nodeContainerList.size(); ++i) {
        //TODO: p2p.SetDeviceAttribute("DataRate", StringValue(GET FROM SNDLIB));
        //TODO: p2p.SetChannelAttribute("Delay", StringValue(GET FROM SNDLIB));
        netDeviceContainerList[i] = p2p.Install(nodeContainerList[i]);
    } 

    std::cout<<"netDeviceContainerList size: "<<netDeviceContainerList.size()<<std::endl;
        
    InternetStackHelper internet;
    //For each node in the input container c, 
    //implement ns3::Ipv4, ns3::Ipv6, ns3::Udp, and, ns3::Tcp classes: 
    internet.Install(c);

    //Assign IP addresses: 
    NS_LOG_INFO("Assign IP Addresses.");
    Ipv4AddressHelper ipv4;
    
    std::vector<Ipv4InterfaceContainer> ipv4List(netDeviceContainerList.size());
    for(unsigned int i=0; i<netDeviceContainerList.size(); ++i){
        std::ostringstream subnet;
        subnet<<"10.1."<<i+1<<".0";
        ipv4.SetBase(subnet.str().c_str(), "255.255.255.0");
        //assign IP address to the net devices specified in the netDeviceContainerList
        //store in a ipv4List 
        ipv4List[i] = ipv4.Assign(netDeviceContainerList[i]);
    }

    std::cout<<"ipv4List size: "<<ipv4List.size()<<std::endl;

    //Build a routing database and initialize the routing tables of the nodes in the simulation.
    //Makes all nodes in the simulation into routers.

//option1: create your own function to call PopulateRoutingTables() so that we can reconfigure BuildGlobalRoutingDatabase()
//option2: find global-route-manager-impl.cc create a custom function and call that rather BuildGlobalRoutingDatabase() 

    //PopulateRoutingTables() calls BuildGlobalRoutingDatabase() and InitializeRoutes()
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
/*
 GlobalRouteManagerImpl::BuildGlobalRoutingDatabase () 
 {
   NS_LOG_FUNCTION (this);
 //
 // Walk the list of nodes looking for the GlobalRouter Interface.  Nodes with
 // global router interfaces are, not too surprisingly, our routers.
 //
   NodeList::Iterator listEnd = NodeList::End ();
   for (NodeList::Iterator i = NodeList::Begin (); i != listEnd; i++)
     {
       Ptr<Node> node = *i;
 
       Ptr<GlobalRouter> rtr = node->GetObject<GlobalRouter> ();
 //
 // Ignore nodes that aren't participating in routing.
 //
       if (!rtr)
         {
           continue;
         }
 //
 // You must call DiscoverLSAs () before trying to use any routing info or to
 // update LSAs.  DiscoverLSAs () drives the process of discovering routes in
 // the GlobalRouter.  Afterward, you may use GetNumLSAs (), which is a very
 // computationally inexpensive call.  If you call GetNumLSAs () before calling 
 // DiscoverLSAs () will get zero as the number since no routes have been 
 // found.
 //
       Ptr<Ipv4GlobalRouting> grouting = rtr->GetRoutingProtocol ();
       uint32_t numLSAs = rtr->DiscoverLSAs ();
       NS_LOG_LOGIC ("Found " << numLSAs << " LSAs");
 
       for (uint32_t j = 0; j < numLSAs; ++j)
         {
           GlobalRoutingLSA* lsa = new GlobalRoutingLSA ();
 //
 // This is the call to actually fetch a Link State Advertisement from the 
 // router.
 //
           rtr->GetLSA (j, *lsa);
           NS_LOG_LOGIC (*lsa);
 //
 // Write the newly discovered link state advertisement to the database.
 //
           m_lsdb->Insert (lsa->GetLinkStateId (), lsa); 
         }
     }
 }

 GlobalRouteManagerImpl::InitializeRoutes ()
 {
   NS_LOG_FUNCTION (this);
 //
 // Walk the list of nodes in the system.
 //
   NS_LOG_INFO ("About to start SPF calculation");
   NodeList::Iterator listEnd = NodeList::End ();
   for (NodeList::Iterator i = NodeList::Begin (); i != listEnd; i++)
     {
       Ptr<Node> node = *i;
 //
 // Look for the GlobalRouter interface that indicates that the node is
 // participating in routing.
 //
       Ptr<GlobalRouter> rtr = 
         node->GetObject<GlobalRouter> ();
 
       uint32_t systemId = MpiInterface::GetSystemId ();
       // Ignore nodes that are not assigned to our systemId (distributed sim)
       if (node->GetSystemId () != systemId) 
         {
           continue;
         }
 
 //
 // if the node has a global router interface, then run the global routing
 // algorithms.
 //
       if (rtr && rtr->GetNumLSAs () )
         {
           SPFCalculate (rtr->GetRouterId ());
         }
     }
   NS_LOG_INFO ("Finished SPF calculation");
 }
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quagga ospf_spf_calculate
 void
 GlobalRouteManagerImpl::SPFCalculate (Ipv4Address root)
 {
   NS_LOG_FUNCTION (this << root);
 
   SPFVertex *v;
 //
 // Initialize the Link State Database.
 //
   m_lsdb->Initialize ();
 //
 // The candidate queue is a priority queue of SPFVertex objects, with the top
 // of the queue being the closest vertex in terms of distance from the root
 // of the tree.  Initially, this queue is empty.
 //
   CandidateQueue candidate;
   NS_ASSERT (candidate.Size () == 0);
 //
 // Initialize the shortest-path tree to only contain the router doing the 
 // calculation.  Each router (and corresponding network) is a vertex in the
 // shortest path first (SPF) tree.
 //
   v = new SPFVertex (m_lsdb->GetLSA (root));
 // 
 // This vertex is the root of the SPF tree and it is distance 0 from the root.
 // We also mark this vertex as being in the SPF tree.
 //
   m_spfroot= v;
   v->SetDistanceFromRoot (0);
   v->GetLSA ()->SetStatus (GlobalRoutingLSA::LSA_SPF_IN_SPFTREE);
   NS_LOG_LOGIC ("Starting SPFCalculate for node " << root);
 
 //
 // Optimize SPF calculation, for ns-3.
 // We do not need to calculate SPF for every node in the network if this
 // node has only one interface through which another router can be 
 // reached.  Instead, short-circuit this computation and just install
 // a default route in the CheckForStubNode() method.
 //
   if (NodeList::GetNNodes () > 0 && CheckForStubNode (root))
     {
       NS_LOG_LOGIC ("SPFCalculate truncated for stub node " << root);
       delete m_spfroot;
       return;
     }
 
   for (;;)
     {
 //
 // The operations we need to do are given in the OSPF RFC which we reference
 // as we go along.
 //
 // RFC2328 16.1. (2). 
 //
 // We examine the Global Router Link Records in the Link State 
 // Advertisements of the current vertex.  If there are any point-to-point
 // links to unexplored adjacent vertices we add them to the tree and update
 // the distance and next hop information on how to get there.  We also add
 // the new vertices to the candidate queue (the priority queue ordered by
 // shortest path).  If the new vertices represent shorter paths, we use them
 // and update the path cost.
 //
       SPFNext (v, candidate);
 //
 // RFC2328 16.1. (3). 
 //
 // If at this step the candidate list is empty, the shortest-path tree (of
 // transit vertices) has been completely built and this stage of the
 // procedure terminates. 
 //
       if (candidate.Size () == 0)
         {
           break;
         }
 //
 // Choose the vertex belonging to the candidate list that is closest to the
 // root, and add it to the shortest-path tree (removing it from the candidate
 // list in the process).
 //
 // Recall that in the previous step, we created SPFVertex structures for each
 // of the routers found in the Global Router Link Records and added tehm to 
 // the candidate list.
 //
       NS_LOG_LOGIC (candidate);
       v = candidate.Pop ();
       NS_LOG_LOGIC ("Popped vertex " << v->GetVertexId ());
 //
 // Update the status field of the vertex to indicate that it is in the SPF
 // tree.
 //
       v->GetLSA ()->SetStatus (GlobalRoutingLSA::LSA_SPF_IN_SPFTREE);
 //
 // The current vertex has a parent pointer.  By calling this rather oddly 
 // named method (blame quagga) we add the current vertex to the list of 
 // children of that parent vertex.  In the next hop calculation called during
 // SPFNext, the parent pointer was set but the vertex has been orphaned up
 // to now.
 //
       SPFVertexAddParent (v);
 //
 // Note that when there is a choice of vertices closest to the root, network
 // vertices must be chosen before router vertices in order to necessarily
 // find all equal-cost paths. 
 //
 // RFC2328 16.1. (4). 
 //
 // This is the method that actually adds the routes.  It'll walk the list
 // of nodes in the system, looking for the node corresponding to the router
 // ID of the root of the tree -- that is the router we're building the routes
 // for.  It looks for the Ipv4 interface of that node and remembers it.  So
 // we are only actually adding routes to that one node at the root of the SPF 
 // tree.
 //
 // We're going to pop of a pointer to every vertex in the tree except the 
 // root in order of distance from the root.  For each of the vertices, we call
 // SPFIntraAddRouter ().  Down in SPFIntraAddRouter, we look at all of the 
 // point-to-point Global Router Link Records (the links to nodes adjacent to
 // the node represented by the vertex).  We add a route to the IP address 
 // specified by the m_linkData field of each of those link records.  This will
 // be the *local* IP address associated with the interface attached to the 
 // link.  We use the outbound interface and next hop information present in 
 // the vertex <v> which have possibly been inherited from the root.
 //
 // To summarize, we're going to look at the node represented by <v> and loop
 // through its point-to-point links, adding a *host* route to the local IP
 // address (at the <v> side) for each of those links.
 //
       if (v->GetVertexType () == SPFVertex::VertexRouter)
         {
           SPFIntraAddRouter (v);
         }
       else if (v->GetVertexType () == SPFVertex::VertexNetwork)
         {
           SPFIntraAddTransit (v);
         }
       else
         {
           NS_ASSERT_MSG (0, "illegal SPFVertex type");
         }
 //
 // RFC2328 16.1. (5). 
 //
 // Iterate the algorithm by returning to Step 2 until there are no more
 // candidate vertices.
 
     }  // end for loop
 
 // Second stage of SPF calculation procedure
   SPFProcessStubs (m_spfroot);
   for (uint32_t i = 0; i < m_lsdb->GetNumExtLSAs (); i++)
     {
       m_spfroot->ClearVertexProcessed ();
       GlobalRoutingLSA *extlsa = m_lsdb->GetExtLSA (i);
       NS_LOG_LOGIC ("Processing External LSA with id " << extlsa->GetLinkStateId ());
       ProcessASExternals (m_spfroot, extlsa);
     }
 
 //
 // We're all done setting the routing information for the node at the root of
 // the SPF tree.  Delete all of the vertices and corresponding resources.  Go
 // possibly do it again for the next router.
 //
   delete m_spfroot;
   m_spfroot = 0;
 }
*/


//////////////////////////////////////////////////////////////////////////////////

    //Create OnOff packet sender appilcation, to the ip address of node 3
    //sent from node 0.
    NS_LOG_INFO("Create Application.");
    uint16_t port = 9;

    OnOffHelper onoff ("ns3::UdpSocketFactory", 
            Address (InetSocketAddress (ipv4List[9].GetAddress (1), port)));
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

//////////////////////////////////////////////////////////////////////////////////

/*
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
    */
    // Flow Monitor
    //Ptr<FlowMonitor> flowMonitor;
    //FlowMonitorHelper flowmonHelper; 
    //if (enableFlowMonitor) {
    //    flowMonitor = flowmonHelper.InstallAll ();
    //}

    //Enable ascii filestream of p2p trace:
    //AsciiTraceHelper ascii;
    //p2p.EnableAsciiAll (ascii.CreateFileStream ("firstsim/FirstSimTrace.tr"));
    //p2p.EnablePcapAll ("firstsim/pcap/FirstSim");

/////////////////////////////////////////////////////////////////////////////////////

    //Create xml animation file and routing table
    AnimationInterface anim("sndlibsim/FirstSimAnimation.xml");
    anim.EnablePacketMetadata(true);
    anim.EnableIpv4RouteTracking("sndlibsim/FirstRoutingTable.xml", 
            Seconds(0), Seconds(10), Seconds(0.25));

    NS_LOG_INFO ("Run Simulation.");

    Simulator::Stop(Seconds(11.0)); //required
    
    Simulator::Run();
    
    //if enableFlowMonitor = true, put flow monitor output in xml
    //if (enableFlowMonitor) {
    //   flowMonitor->SerializeToXmlFile ("firstsim/FirstSimFlowMon.xml", true, true);
    //}

    Simulator::Destroy();
    NS_LOG_INFO ("Done.");

    return 0;
}
