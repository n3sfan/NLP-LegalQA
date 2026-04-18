/**
 * Mock API layer — mirrors exact Python schemas.
 * Replace imports in hooks with ./client.ts once FastAPI pipeline endpoints exist.
 */

import type {
  SearchResult,
  RerankResult,
  DecomposeResult,
  SearchApiResponse,
  ArticleApiResponse,
} from './types';

// ── Helpers ────────────────────────────────────────────────────────────────────

function delay(ms: number) {
  return new Promise<void>((r) => setTimeout(r, ms));
}

// ── Decompose mock ─────────────────────────────────────────────────────────────

const DECOMPOSITIONS: Record<string, DecomposeResult> = {
  'không đội mũ bảo hiểm phạt bao nhiêu': {
    success: true,
    reasoning:
      'Truy vấn hỏi về mức phạt khi không đội mũ bảo hiểm. Tôi tách thành 2 truy vấn con: (1) mức phạt cụ thể khi không đội mũ bảo hiểm, (2) quy định bắt buộc đội mũ bảo hiểm để xác định đối tượng áp dụng.',
    sub_queries: [
      { query: 'mức phạt không đội mũ bảo hiểm xe máy', index: 0 },
      { query: 'quy định bắt buộc đội mũ bảo hiểm xe máy Việt Nam', index: 1 },
    ],
  },
  'xe chạy quá tốc độ phạt bao nhiêu': {
    success: true,
    reasoning:
      'Người dùng hỏi về mức phạt khi xe chạy quá tốc độ. Tôi tách thành 2 truy vấn con: (1) mức phạt vượt quá tốc độ cho phép, (2) quy định về giới hạn tốc độ trên đường bộ.',
    sub_queries: [
      { query: 'mức phạt xe chạy quá tốc độ cho phép', index: 0 },
      { query: 'quy định giới hạn tốc độ xe cơ giới đường bộ', index: 1 },
    ],
  },
  'chạy xe đêm phạt không': {
    success: true,
    reasoning:
      'Hỏi về quy định điều kiện chạy xe ban đêm và các lỗi liên quan. Tôi tách thành 2 truy vấn con: (1) quy định đèn chiếu sáng khi chạy xe đêm, (2) các lỗi thường gặp khi chạy xe vào ban đêm.',
    sub_queries: [
      { query: 'quy định đèn chiếu sáng xe máy ban đêm', index: 0 },
      { query: 'lỗi thường gặp khi chạy xe đêm và mức phạt', index: 1 },
    ],
  },
};

// ── Vector search mock ─────────────────────────────────────────────────────────

const VECTOR_RESULTS: Record<string, SearchResult[]> = {
  'không đội mũ bảo hiểm phạt bao nhiêu': [
    { uid: '56/2024/QH15::article::10::clause::1::point::a', label: 'Point',    score: 0.94 },
    { uid: '56/2024/QH15::article::10::clause::1',            label: 'Clause',   score: 0.91 },
    { uid: '56/2024/QH15::article::10',                        label: 'Article', score: 0.88 },
    { uid: '56/2024/QH15::article::8',                         label: 'Article', score: 0.72 },
  ],
  'xe chạy quá tốc độ phạt bao nhiêu': [
    { uid: '56/2024/QH15::article::5::clause::2::point::a', label: 'Point',    score: 0.93 },
    { uid: '56/2024/QH15::article::5::clause::2',           label: 'Clause',   score: 0.90 },
    { uid: '56/2024/QH15::article::5',                       label: 'Article', score: 0.87 },
  ],
  'chạy xe đêm phạt không': [
    { uid: '56/2024/QH15::article::12::clause::1::point::b', label: 'Point',    score: 0.91 },
    { uid: '56/2024/QH15::article::12::clause::1',           label: 'Clause',   score: 0.89 },
    { uid: '56/2024/QH15::article::12',                       label: 'Article', score: 0.85 },
  ],
};

// ── Rerank mock ────────────────────────────────────────────────────────────────

const RERANK_TEXTS: Record<string, RerankResult[]> = {
  'không đội mũ bảo hiểm phạt bao nhiêu': [
    {
      uid: '56/2024/QH15::article::10::clause::1::point::a',
      label: 'Point',
      score: 0.94,
      text: 'a) Không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng cách khi tham gia giao thông đường bộ.',
      rerank_score: 0.96,
    },
    {
      uid: '56/2024/QH15::article::10::clause::1',
      label: 'Clause',
      score: 0.91,
      text: '1. Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe mô tô, xe gắn máy (kể cả xe máy điện) thực hiện một trong các hành vi vi phạm sau đây: a) Không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng cách khi tham gia giao thông đường bộ.',
      rerank_score: 0.93,
    },
    {
      uid: '56/2024/QH15::article::10',
      label: 'Article',
      score: 0.88,
      text: 'Điều 10. Mức phạt tiền đối với hành vi vi phạm quy định về đội mũ bảo hiểm đối với người điều khiển xe mô tô, xe gắn máy.\n1. Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe mô tô, xe gắn máy không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng cách.',
      rerank_score: 0.90,
    },
    {
      uid: '56/2024/QH15::article::8',
      label: 'Article',
      score: 0.72,
      text: 'Điều 8. Quy định về trang bị phương tiện tham gia giao thông. Các phương tiện giao thông đường bộ phải được trang bị đầy đủ các thiết bị an toàn theo quy định của pháp luật.',
      rerank_score: 0.74,
    },
  ],
  'xe chạy quá tốc độ phạt bao nhiêu': [
    {
      uid: '56/2024/QH15::article::5::clause::2::point::a',
      label: 'Point',
      score: 0.93,
      text: 'a) Xe chạy quá tốc độ từ 05 km/h đến dưới 10 km/h so với tốc độ cho phép.',
      rerank_score: 0.95,
    },
    {
      uid: '56/2024/QH15::article::5::clause::2',
      label: 'Clause',
      score: 0.90,
      text: '2. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe chạy quá tốc độ cho phép từ 05 km/h đến dưới 10 km/h.',
      rerank_score: 0.92,
    },
    {
      uid: '56/2024/QH15::article::5',
      label: 'Article',
      score: 0.87,
      text: 'Điều 5. Xử phạt vi phạm quy định về tốc độ xe cơ giới đường bộ. 1. Phạt tiền từ 200.000 đồng đến 400.000 đồng đối với người điều khiển xe chạy quá tốc độ cho phép dưới 05 km/h. 2. Phạt tiền từ 400.000 đồng đến 600.000 đồng đối với người điều khiển xe chạy quá tốc độ cho phép từ 05 km/h đến dưới 10 km/h.',
      rerank_score: 0.89,
    },
  ],
  'chạy xe đêm phạt không': [
    {
      uid: '56/2024/QH15::article::12::clause::1::point::b',
      label: 'Point',
      score: 0.91,
      text: 'b) Xe không có đèn chiếu sáng hoặc đèn chiếu sáng không đủ tiêu chuẩn kỹ thuật khi tham gia giao thông vào ban đêm.',
      rerank_score: 0.93,
    },
    {
      uid: '56/2024/QH15::article::12::clause::1',
      label: 'Clause',
      score: 0.89,
      text: '1. Phạt tiền từ 300.000 đồng đến 500.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi vi phạm sau đây: b) Xe không có đèn chiếu sáng hoặc đèn chiếu sáng không đủ tiêu chuẩn kỹ thuật khi tham gia giao thông vào ban đêm.',
      rerank_score: 0.91,
    },
    {
      uid: '56/2024/QH15::article::12',
      label: 'Article',
      score: 0.85,
      text: 'Điều 12. Xử phạt vi phạm quy định về điều kiện chạy xe vào ban đêm. 1. Phạt tiền từ 300.000 đồng đến 500.000 đồng đối với người điều khiển xe không có đèn chiếu sáng hoặc đèn chiếu sáng không đủ tiêu chuẩn kỹ thuật khi tham gia giao thông vào ban đêm.',
      rerank_score: 0.88,
    },
  ],
};

// ── Public mock API ─────────────────────────────────────────────────────────────

export async function mockDecomposeQuery(query: string): Promise<DecomposeResult> {
  await delay(800);
  return (
    DECOMPOSITIONS[query] ?? {
      success: true,
      reasoning: `Phân tích truy vấn: "${query}". Truy vấn này liên quan đến quy định pháp luật giao thông đường bộ Việt Nam. Tôi sẽ tìm kiếm các điều khoản liên quan đến nội dung này.`,
      sub_queries: [
        { query, index: 0 },
        { query: `${query} mức phạt`, index: 1 },
      ],
    }
  );
}

export async function mockVectorSearch(subQueries: string[]): Promise<SearchResult[]> {
  await delay(1200);
  // Match by the first sub-query's root
  const first = subQueries[0]?.toLowerCase() ?? '';
  for (const key of Object.keys(VECTOR_RESULTS)) {
    if (first.includes(key.split(' ')[0])) {
      return VECTOR_RESULTS[key];
    }
  }
  return VECTOR_RESULTS['không đội mũ bảo hiểm phạt bao nhiêu']!;
}

export async function mockRerank(
  _subQueries: string[],
  vectorResults: SearchResult[],
): Promise<RerankResult[]> {
  await delay(600);
  const templates = RERANK_TEXTS['không đội mũ bảo hiểm phạt bao nhiêu'];
  return vectorResults.map((vr, i) => {
    const template = templates[i] ?? templates[0]!;
    return { ...template, uid: vr.uid, label: vr.label, score: vr.score };
  });
}

// Existing keyword search (for HistoryPage)
export async function mockSearchKeyword(_q: string): Promise<SearchApiResponse> {
  await delay(400);
  return {
    articles: [
      {
        type: 'article',
        doc_identity: '56/2024/QH15',
        doc_name: 'Luật Trật tự, an toàn giao thông đường bộ năm 2024',
        article_num: 10,
        title: 'Mức phạt tiền đối với hành vi vi phạm quy định về đội mũ bảo hiểm',
        uid: '56/2024/QH15::article::10',
      },
      {
        type: 'article',
        doc_identity: '56/2024/QH15',
        doc_name: 'Luật Trật tự, an toàn giao thông đường bộ năm 2024',
        article_num: 8,
        title: 'Quy định về trang bị phương tiện tham gia giao thông',
        uid: '56/2024/QH15::article::8',
      },
    ],
    clauses: [
      {
        type: 'clause',
        doc_identity: '56/2024/QH15',
        article_num: 10,
        clause_num: 1,
        content: 'Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe mô tô, xe gắn máy không đội mũ bảo hiểm.',
        uid: '56/2024/QH15::article::10::clause::1',
      },
    ],
    points: [
      {
        type: 'point',
        doc_identity: '56/2024/QH15',
        article_num: 10,
        clause_num: 1,
        point_letter: 'a',
        content: 'Không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng cách khi tham gia giao thông đường bộ.',
        uid: '56/2024/QH15::article::10::clause::1::point::a',
      },
    ],
  };
}

export async function mockGetArticle(doc_identity: string, article_num: number): Promise<ArticleApiResponse> {
  await delay(300);
  if (doc_identity !== '56/2024/QH15') {
    throw new Error(`Article not found: ${doc_identity}/${article_num}`);
  }
  return {
    doc_identity,
    article_num,
    article_uid: `${doc_identity}::article::${article_num}`,
    clauses: [
      {
        type: 'clause',
        doc_identity,
        article_num,
        clause_num: 1,
        content: 'Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe mô tô, xe gắn máy (kể cả xe máy điện) không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng cách khi tham gia giao thông đường bộ.',
        uid: `${doc_identity}::article::${article_num}::clause::1`,
      },
      {
        type: 'clause',
        doc_identity,
        article_num,
        clause_num: 2,
        content: 'Phạt tiền từ 200.000 đồng đến 300.000 đồng đối với người điều khiển xe mô tô, xe gắn máy đội mũ bảo hiểm quấn bên ngoài quai đăng ten.',
        uid: `${doc_identity}::article::${article_num}::clause::2`,
      },
    ],
    points: [
      {
        type: 'point',
        doc_identity,
        article_num,
        clause_num: 1,
        point_letter: 'a',
        content: 'Không đội mũ bảo hiểm hoặc đội mũ bảo hiểm không cài quai đúng cách khi tham gia giao thông đường bộ.',
        uid: `${doc_identity}::article::${article_num}::clause::1::point::a`,
      },
      {
        type: 'point',
        doc_identity,
        article_num,
        clause_num: 1,
        point_letter: 'b',
        content: 'Đội mũ bảo hiểm không cài quai đúng cách.',
        uid: `${doc_identity}::article::${article_num}::clause::1::point::b`,
      },
    ],
  };
}
