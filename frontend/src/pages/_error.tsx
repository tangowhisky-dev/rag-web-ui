import type { NextPageContext } from "next";

export default function ErrorPage({ statusCode }: { statusCode: number }) {
  return (
    <div style={{ textAlign: "center", padding: "4rem 1rem" }}>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 600 }}>{statusCode}</h1>
      <p style={{ color: "#666" }}>
        {statusCode === 404 ? "Page not found" : "Internal server error"}
      </p>
    </div>
  );
}

ErrorPage.getInitialProps = ({ res, err }: NextPageContext) => {
  const statusCode = res ? res.statusCode : err ? err.statusCode : 404;
  return { statusCode };
};
