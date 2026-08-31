import { CircleAlert, ChevronLeft, ChevronRight } from "lucide-react";
import { CSSProperties, ReactNode, useEffect, useMemo, useState } from "react";

const DEFAULT_PAGE_SIZE = 5;

export function HelpTip({ text }: { text: string }) {
  return (
    <span className="help-tip-wrap">
      <button aria-label={text} className="help-tip" title={text} type="button">
        <CircleAlert aria-hidden="true" size={14} />
      </button>
      <span className="help-tip-content" role="tooltip">{text}</span>
    </span>
  );
}

export function LabelWithHelp({ children, help }: { children: ReactNode; help: string }) {
  return <span className="label label-with-help">{children}<HelpTip text={help} /></span>;
}

export function ActionWithHelp({ children, help }: { children: ReactNode; help: string }) {
  return <span className="action-with-help">{children}<HelpTip text={help} /></span>;
}

export function PaginatedTable<T>({
  rows,
  headers,
  renderRow,
  pageSize = DEFAULT_PAGE_SIZE,
  className = "",
  style
}: {
  rows: T[];
  headers: ReactNode[];
  renderRow: (row: T, index: number) => ReactNode;
  pageSize?: number;
  className?: string;
  style?: CSSProperties;
}) {
  const [page, setPage] = useState(1);
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));

  useEffect(() => {
    setPage((current) => Math.min(current, pageCount));
  }, [pageCount]);

  const visibleRows = useMemo(() => {
    const start = (page - 1) * pageSize;
    return rows.slice(start, start + pageSize);
  }, [page, pageSize, rows]);
  const firstRow = rows.length ? (page - 1) * pageSize + 1 : 0;
  const lastRow = Math.min(page * pageSize, rows.length);

  return (
    <div className="paginated-table" style={style}>
      <div className={`table-wrap paginated-table-scroll ${className}`.trim()}>
        <table className="table">
          <thead><tr>{headers.map((header, index) => <th key={index}>{header}</th>)}</tr></thead>
          <tbody>{visibleRows.map((row, index) => renderRow(row, (page - 1) * pageSize + index))}</tbody>
        </table>
      </div>
      <div className="table-pagination" aria-label="Table pagination">
        <span>{firstRow}-{lastRow} of {rows.length}</span>
        <span>Page {page} of {pageCount}</span>
        <div className="table-pagination-actions">
          <button aria-label="Previous page" className="button icon-button" disabled={page <= 1} onClick={() => setPage((current) => Math.max(1, current - 1))} title="Previous page" type="button">
            <ChevronLeft size={16} />
          </button>
          <button aria-label="Next page" className="button icon-button" disabled={page >= pageCount} onClick={() => setPage((current) => Math.min(pageCount, current + 1))} title="Next page" type="button">
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
