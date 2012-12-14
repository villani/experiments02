package villani.eti.br;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import java.util.TreeMap;

import javax.imageio.ImageIO;

import mulan.data.MultiLabelInstances;
import net.semanticmetadata.lire.imageanalysis.mpeg7.EdgeHistogramImplementation;

public class Caracteristicas {

	private static LogBuilder log;
	private static TreeMap<String, String> entradas;
	private static File folder;
	private static String dataset;
	private static String csv;
	private static String txt;

	public static void setLog(LogBuilder log) {
		Caracteristicas.log = log;
	}

	public static void setEntradas(TreeMap<String, String> entradas) {
		Caracteristicas.entradas = entradas;
		folder = new File(Caracteristicas.entradas.get("folder"));
		dataset = Caracteristicas.entradas.get("dataset");
		csv = Caracteristicas.entradas.get("csv");
		txt = Caracteristicas.entradas.get("txt");
	}

	public static MultiLabelInstances obtemEHD() {

		log.write("Construindo conjunto de instancias EHD");
		MultiLabelInstances instanciasML = null;

		log.write("- Obtendo o conjunto de rótulos IRMA");
		XmlIrmaCodeBuilder xicb;
		try {
			log.write("- Criando arquivo xml com a estrutura de códigos IRMA");
			xicb = new XmlIrmaCodeBuilder(txt, dataset);
			if (xicb.hasXml())
				log.write("- Arquivo xml com a estrutura de código IRMA criado com êxito");
		} catch (IOException e) {
			log.write("- Falha ao obter relação nome da imagem/ código IRMA: "
					+ e.getMessage());
			System.exit(0);
		}

		log.write("- Definindo atributos do conjunto");
		try {
			RelationBuilder instanciasEHD = new RelationBuilder(dataset);
			for (int i = 0; i < 80; i++)
				instanciasEHD.defineAttribute("ehd" + i, "numeric");

			log.write("- Salvando a lista de atributos e incluindo a lista de rótulos a partir do xml");
			instanciasEHD.saveAttributes();

			log.write("Armazenando no conjunto as amostras com características EHD");
			try {
				log.write(" - Obtendo a relação nome da imagem/código IRMA do arquivo: " + csv);
				File relacaoImagemCodigo = new File(csv);
				TreeMap<String, String> relacao = new TreeMap<String, String>();
				Scanner leitor01 = new Scanner(relacaoImagemCodigo);
				while (leitor01.hasNextLine()) {
					String[] campos = leitor01.nextLine().split(";");
					relacao.put(campos[0], campos[1]);
				}
				leitor01.close();

				log.write("- Criando objeto que converte o codigo IRMA para binário e que também necessita do xml criado anteriormente");
				IrmaCode conversor = new IrmaCode(dataset);

				log.write("- Obtendo características EHD para cada imagem");
				File[] imagens = folder.listFiles();
				for (File imagem : imagens) {
					//if(!imagem.canRead()) System.out.println(imagem.getAbsolutePath());

					// Construo o extrator e forneço a imagem para obter características
					EdgeHistogramImplementation extrator = new EdgeHistogramImplementation(ImageIO.read(imagem));

					// Obtenho o histograma de bordas referente
					int[] ehd = extrator.setEdgeHistogram();

					// Crio uma amostra para armazenar as características obtidas
					String amostra = "";
					for (int e : ehd) amostra += e + ",";

					// Armazeno o respectivo rótulo IRMA binário à amostra
					String nomeImg = imagem.getName().split("\\.")[0];
					amostra += conversor.toBinary(relacao.get(nomeImg));

					// Armazeno a amostra no conjunto de dados
					instanciasEHD.insertData(amostra);
				}
				
				log.write("Novo conjunto de amostras salvo em: " + dataset + ".arff");
				instanciasML = instanciasEHD.saveRelation(); // armazenando o retorno do método

			} catch (FileNotFoundException e) {
				log.write("Um arquivo não pode ser encontrado: " + e.getMessage());
				System.exit(0);
			} catch (IOException e) {
				log.write("Falha ao ler uma imagem: " + e.getMessage());
				System.exit(0);
			} catch (Exception e) {
				log.write("Falha ao inserir a amostra: " + e.getMessage());
				System.exit(0);
			}

		} catch (Exception e) {
			log.write("Falha ao construir conjunto de instancias EHD: "
					+ e.getMessage());
			System.exit(0);
		}
		
		return instanciasML;
	}
}