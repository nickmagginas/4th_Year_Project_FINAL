#include "parser.h"

std :: map <std :: string , std :: tuple <std :: string, std :: string , std :: string, std :: string>> Parser :: get_links(){
	return links;
}

std :: map <std :: string , std :: tuple<std :: string, std :: string>> Parser :: get_nodes(){
	return nodes;
}	

std :: vector <std :: string> Parser :: tokenize(std::string line){
	std :: vector <std :: string> tokens;
	std :: stringstream check1(line); 
    std :: string intermediate;
    while(getline(check1, intermediate, ' ')) 
    { 
        tokens.push_back(intermediate); 
    } 
    return tokens;
} 

void Parser :: read_file()
{	
	std :: cout << "Hi" << std :: endl;
	std :: string filename = "germany50.txt";
	std :: tuple <std :: string, std :: string , std :: string, std :: string> link_data;
	std :: tuple <std :: string, std :: string> coordinates;
	std :: map <std :: string,int> section_map;
	std :: string line;
	section_map["NODES"] = 1;
	section_map["LINKS"] = 2;
	section_map["DEMANDS"] = 3;
	std :: string section;
	int current = 0;
	std :: cout << "Parsing File " << filename << std :: endl;
	std :: ifstream file;
	file.open(filename);
	if (!file){
		std :: cerr << "Cannot Open FIle" << std :: endl;
		exit(1);
	}
	while(file){
		std :: getline(file,line);
		std :: vector<std :: string> v = tokenize(line);
		if (v.size()!=0){
			section = v[0];
		}
		else
		{
			continue;
		}
		switch (section_map[section]){
			case 1:
				current = 1;
				break;
			case 2:
				current = 2;
				break;
			case 3:
				current = 3;
				break;
		}
		switch (current){
			case 1:
				if (v.size() == 7){
					std :: get<0>(coordinates) = v[4];
					std :: get<1>(coordinates) = v[5];
					nodes[v[2]] = coordinates; 
				}
				break;
			case 2:
				if (v.size() == 15){
					std :: get<0>(link_data) = v[4];
					std :: get<1>(link_data) = v[5];
					std :: get<2>(link_data) = v[12];
					std :: get<3>(link_data) = v[13];
					links[v[2]] = link_data;
				}
				break;
			default:
				break;

		}
	}
}