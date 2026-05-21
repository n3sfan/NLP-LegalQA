from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from openai import OpenAI

node_properties_query = """
MATCH (n)
WITH labels(n)[0] AS label, keys(n) AS propertyNames
UNWIND propertyNames AS prop
WITH label, collect(DISTINCT prop) AS props
WHERE label IS NOT NULL
RETURN {labels: label, properties: props} AS output
"""

rel_properties_query = """
MATCH ()-[r]->()
WITH type(r) AS relType, keys(r) AS propertyNames
UNWIND (case when propertyNames = [] then [null] else propertyNames end) AS prop
WITH relType, collect(DISTINCT prop) AS rawProps
RETURN {type: relType, properties: [p IN rawProps WHERE p IS NOT NULL]} AS output
"""

rel_query = """
CALL db.schema.visualization() 
YIELD nodes, relationships
UNWIND relationships AS rel
WITH startNode(rel) AS source, endNode(rel) AS target, type(rel) AS relType
RETURN {source: labels(source)[0], relationship: relType, target: labels(target)[0]} AS output
"""

def schema_text(node_props, rel_props, rels):
    return f"""
  Đây là schema của cơ sở dữ liệu Neo4j.
  Các thuộc tính của node bao gồm:
  {node_props}
  Các thuộc tính quan hệ (relationship) bao gồm:
  {rel_props}
  Hướng quan hệ từ source node đến target nodes như sau:
  {rels}
  Hãy đảm bảo tuân thủ đúng các loại quan hệ và hướng của chúng.
  """

class Neo4jGeminiQuery:
    def __init__(self, url, user, password, gemini_api_key):
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=gemini_api_key
        )
        # construct schema
        self.schema = self.generate_schema()


    def generate_schema(self):
        node_props = self.query_database(node_properties_query)
        rel_props = self.query_database(rel_properties_query)
        rels = self.query_database(rel_query)
        return schema_text(node_props, rel_props, rels)

    def refresh_schema(self):
        self.schema = self.generate_schema()

    def get_rewrite_message(self):
        return f"""
        Bạn là chuyên gia xử lý ngôn ngữ tự nhiên cho các câu hỏi truy vấn đến cơ sở dữ liệu đồ thị Neo4j.
        Nhiệm vụ: Dựa vào câu hỏi dưới dạng ngôn ngữ tự nhiên cho người dùng, hãy viết lại câu hỏi thành câu truy vấn dưới dạng ngôn ngữ tự nhiên dựa theo schema được định nghĩa mà từ câu truy vấn đó có thể chuyển sang câu lệnh Cypher một cách chính xác.
        Hướng dẫn các nguyên tắc:
        - Câu truy vấn phải giữ đúng nội dung của câu hỏi, không được tự phát sinh làm thay đổi các thông tin về node và thuộc tính có trong câu hỏi.
        - Trong câu truy vấn, tuyệt đối không được nhắc đến tên loại quan hệ (relationship type) cụ thể.
        - Dựa vào schema, bạn phải sắp xếp lại trật tự trong câu truy vấn để tuân theo hướng của các quan hệ, sao cho source node xuất hiện trước target node.
        - Đối với các thuộc tính dùng để định danh, hãy luôn sử dụng từ "bằng" kết hợp với giá trị tìm kiếm dưới dạng chữ (ví dụ: a, b, c), số (ví dụ: 1, 2, 3) hay chữ số La Mã (ví dụ: I, II, III), đồng thời giữ đúng tên của thuộc tính định danh đó.
        - Đối với các thuộc tính văn bản mô tả, hãy luôn sử dụng từ "chứa" kết hợp với giá trị tìm kiếm.
        Dưới đây là schema của cơ sở dữ liệu Neo4j:
        {self.schema}
        Quy tắc đầu ra:
        - Chỉ trả về duy nhất câu truy vấn hợp lệ dưới dạng ngôn ngữ tự nhiên.
        - Tuyệt đối không thêm bất kỳ văn bản chào hỏi, giải thích, lưu ý hay phân tích nào khác.
        """

    def rewrite_question(self, question):
        messages = [
            {"role": "system", "content": self.get_rewrite_message()},
            {"role": "user", "content": question},
        ]

        completions = self.client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0.0,
            # max_tokens=1000,
            messages=messages
        )
        return completions.choices[0].message.content

    def get_system_message(self):
        return f"""
        Bạn là chuyên gia viết các câu truy vấn Cypher cho cơ sở dữ liệu đồ thị Neo4j, đóng vai trò tiếp nhận câu hỏi dưới dạng ngôn ngữ tự nhiên và tạo câu lệnh Cypher tương ứng với câu hỏi đó (Text-to-Cypher).
        Nhiệm vụ: Tạo câu lệnh Cypher để truy vấn cơ sở dữ liệu đồ thị Neo4j dựa trên định nghĩa schema được cung cấp.
        Hướng dẫn các nguyên tắc:
        - Phải sử dụng mối quan hệ có độ dài biến đổi (ví dụ: (A)-[*1..]->(B)) nếu xuất hiện bất kỳ đường đi hợp lệ nào có thể kết nối từ source node đến target node theo định nghĩa trong schema. Bạn tuyệt đối không được sử dụng bất kỳ loại quan hệ (relationship type) cụ thể nào ở mệnh đề MATCH.
        - Theo mặc định, không thêm bất kỳ node trung gian nào vào đường đi kết nối các node trong mệnh đề MATCH nếu node trung gian đó không có trên câu hỏi. Tuy nhiên, nếu câu hỏi của người dùng có yêu cầu sử dụng các thuộc tính của node trung gian để lọc hay trả về thì đến lúc đó bạn phải khai báo rõ ràng node trung gian đó ở đường đi trong mệnh để MATCH để áp dụng mệnh đề WHERE hay RETURN một cách chính xác.
        - Tuyệt đối tuân thủ hướng của các mối quan hệ mà chỉ được định nghĩa ở trong schema. Bạn tuyệt đối không được đảo ngược hướng của quan hệ hay đảo target node lên vị trí của source node trong mệnh đề MATCH ở bất kỳ trường hợp nào.
        - Đối với các thuộc tính dùng để định danh (ví dụ: number), hãy luôn sử dụng toán tử =, ngay cả khi giá trị tìm kiếm là chữ cái (ví dụ: a, b, c), số (ví dụ: 1, 2, 3) hay là chữ số La Mã (ví dụ: I, II, III).
        - Đối với các thuộc tính văn bản mô tả (ví dụ: doc_name, title, content), hãy luôn sử dụng toán tử toán tử CONTAINS trong mệnh đề WHERE, đồng thời áp dụng hàm toLower() cho cả thuộc tính văn bản mô tả đó lẫn giá trị tìm kiếm trong mệnh đề WHERE.
        - Không bao giờ trả về toàn bộ node object. Thay vào đó, ở mệnh để RETURN, chỉ chọn các thuộc tính có liên quan đến câu hỏi của người dùng có trong các node này.
        - Nếu không thể tạo ra câu lệnh Cypher từ schema đã cung cấp, hãy giải thích lý do cho người dùng.
        Dưới đây là schema của cơ sở dữ liệu đồ thị Neo4j:
        {self.schema}
        Quy tắc đầu ra:
        - Chỉ trả về duy nhất câu lệnh Cypher hợp lệ và tuyệt đối không được bọc trong block code markdown (không dùng ```cypher ```).
        - Tuyệt đối không thêm bất kỳ văn bản chào hỏi, giải thích, lưu ý hay phân tích nào khác.
        - Trường hợp duy nhất được phép trả về văn bản mà không có Cypher query là khi không thể tạo được câu lệnh từ schema, lúc đó hãy đưa ra nguyên nhân trực tiếp.
        Ví dụ:
        Input: Ai là người ký Luật đường bộ?
        Output: MATCH (d:Document)-[*1..]->(s:Signer) WHERE toLower(d.doc_name) CONTAINS toLower('Luật đường bộ') RETURN s.name
        Input: Nghị định 168 có bao nhiêu điều?
        Output: MATCH (d:Document)-[*1..]->(a:Article) WHERE toLower(d.doc_name) CONTAINS toLower('Nghị định 168') RETURN count(a)
        Input: Văn bản nào có chứa quy định về việc tịch thu phương tiện?
        Output: MATCH (d:Document)-[*1..]->(child) WHERE child:Section OR child:Part OR child:Chapter OR child:Article OR child:Clause OR child:Point WITH d, child WHERE toLower(child.title) CONTAINS "tịch thu phương tiện" OR toLower(child.content) CONTAINS "tịch thu phương tiện" RETURN DISTINCT d.doc_name

        Lưu ý: Không bao gồm bất kỳ lời giải thích hay xin lỗi nào trong câu trả lời của bạn.
        """

    def query_database(self, neo4j_query, params={}):
        with self.driver.session() as session:
            result = session.run(neo4j_query, params)
            output = [r.values() for r in result]
            output.insert(0, result.keys())
            return output

    def construct_cypher(self, question, rewrite, history=None):
        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": f"""Câu hỏi gốc của người dùng: {question}.
                                Câu truy vấn đã được viết lại: {rewrite}."""},
        ]
        # Used for Cypher healing flows
        if history:
            messages.extend(history)

        completions = self.client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0.0,
            # max_tokens=1000,
            messages=messages
        )
        return completions.choices[0].message.content

    def run(self, question, history=None, retry=True):
        # Construct Cypher statement
        rewrite = self.rewrite_question(question)
        cypher = self.construct_cypher(question, rewrite, history)
        try:
            return self.query_database(cypher)
        # Self-healing flow
        except CypherSyntaxError as e:
            # If out of retries
            if not retry:
              return "Cú pháp Cypher sinh ra từ câu hỏi không hợp lệ."
        # Self-healing Cypher flow by
        # providing specific error to Gemini
            return self.run(
                question,
                [
                    {"role": "assistant", "content": cypher},
                    {
                        "role": "user",
                        "content": f"""Câu truy vấn này trả về lỗi sau: {str(e)} 
                        Hãy cung cấp cho tôi một câu truy vấn đã được cải thiện để chạy được mà không cần bất kỳ lời giải thích hay xin lỗi nào.""",
                    },
                ],
                retry=False
            )