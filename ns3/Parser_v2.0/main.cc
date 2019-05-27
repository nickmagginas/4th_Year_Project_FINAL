#include "parser.h"

int main(int argc, char const *argv[])
{
	Parser g50;
    g50.read_file();
	std :: map <std :: string , std :: tuple <std :: string, std :: string>> nodes;
	nodes = g50.get_nodes();
	std :: map <std :: string , std :: tuple <std :: string, std :: string>> :: iterator it;
	for (it = nodes.begin(); it != nodes.end(); ++it)
	{
		std :: cout << it -> first << std :: endl;
	}
	return 0;
}
